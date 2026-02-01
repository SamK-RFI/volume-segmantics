"""
Model creation, loading, and optimization setup for the 2D trainer.

This module contains:
- Model creation and MeanTeacher wrapping
- Model freezing/unfreezing
- Optimizer and scheduler creation
- Weight loading utilities
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torchvision import models

from volume_segmantics.model.model_2d import create_model_on_device
from volume_segmantics.model.mean_teacher import MeanTeacherModel
from volume_segmantics.model.sam import SAM
from volume_segmantics.utilities.early_stopping import EarlyStopping
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg


class ModelManager:
    """
    Manages model creation, loading, and optimization setup.
    
    Handles:
    - Model creation with optional MeanTeacher wrapping
    - Model freezing/unfreezing
    - Optimizer and scheduler creation
    - Weight loading from checkpoints
    """
    
    def __init__(
        self,
        settings: SimpleNamespace,
        model_device_num: int,
        label_no: int,
    ):
        """
        Initialize model manager.
        
        Args:
            settings: Training settings
            model_device_num: Device number for model
            label_no: Number of classes
        """
        self.settings = settings
        self.model_device_num = model_device_num
        self.label_no = label_no
        self.use_semi_supervised = getattr(settings, "use_semi_supervised", False)
        self.use_sam = getattr(settings, "use_sam", False)
        self.adaptive_sam = getattr(settings, "adaptive_sam", False)
        if self.use_semi_supervised:
            self.ema_decay = getattr(settings, "ema_decay", 0.99)
    
    def get_model_structure_dict(self, settings: SimpleNamespace) -> dict:
        """
        Get model structure dictionary from settings.
        
        Args:
            settings: Training settings
        
        Returns:
            Model structure dictionary
        """
        model_struc_dict = settings.model.copy() if hasattr(settings.model, 'copy') else dict(settings.model)
        
        use_multitask = getattr(settings, "use_multitask", False)
        if use_multitask:
            original_model_type = utils.get_model_type(settings)
            if original_model_type != utils.ModelType.MULTITASK_UNET:
                logging.warning(
                    f"use_multitask enabled but model type is '{original_model_type.name}'. "
                    f"Automatically changing to 'MULTITASK_UNET'."
                )
            model_type = utils.ModelType.MULTITASK_UNET
            logging.info("Multi-task learning enabled - using MultitaskUnet model")
        else:
            model_type = utils.get_model_type(settings)
        
        model_struc_dict["type"] = model_type
        model_struc_dict["in_channels"] = cfg.get_model_input_channels(settings)
        model_struc_dict["classes"] = self.label_no
        
        if use_multitask:
            num_tasks = getattr(settings, "num_tasks", 2)
            decoder_sharing = getattr(settings, "decoder_sharing", "shared")
            
            # Configure output channels per task
            # Task 1 (seg): num_classes, Task 2+ (boundary, etc.): 1 channel each
            task_out_channels = [self.label_no]  # Segmentation
            for i in range(1, num_tasks):
                task_out_channels.append(1)  # Binary output for auxiliary tasks
            
            model_struc_dict["num_tasks"] = num_tasks
            model_struc_dict["decoder_sharing"] = decoder_sharing
            model_struc_dict["task_out_channels"] = task_out_channels
            
            logging.info(f"Multi-task config: {num_tasks} tasks, output channels: {task_out_channels}")
        
        return model_struc_dict
    
    def create_model_and_optimizer(
        self,
        model_struc_dict: dict,
        learning_rate: float,
        frozen: bool = False,
        use_semi_supervised: bool = None,
    ) -> tuple:
        """
        Create model and optimizer.
        
        Args:
            model_struc_dict: Model structure dictionary
            learning_rate: Learning rate for optimizer
            frozen: Whether to freeze encoder
            use_semi_supervised: Whether to use semi-supervised learning (overrides self.use_semi_supervised)
        
        Returns:
            Tuple of (model, optimizer)
        """
        if use_semi_supervised is None:
            use_semi_supervised = self.use_semi_supervised
        
        logging.info(f"Setting up the model on device {self.settings.cuda_device}.")
        base_model = create_model_on_device(
            self.model_device_num, model_struc_dict
        )
        
        # Wrap with MeanTeacherModel if using semi-supervised learning
        # Note: create_model_on_device may wrap with DataParallel, so we need to handle that
        if use_semi_supervised:
            # Unwrap DataParallel if present
            if isinstance(base_model, DataParallel):
                base_model_unwrapped = base_model.module
                was_dataparallel = True
            else:
                base_model_unwrapped = base_model
                was_dataparallel = False
            
            # Wrap with MeanTeacherModel
            model = MeanTeacherModel(
                student_model=base_model_unwrapped,
                ema_decay=self.ema_decay
            )
            
            # Re-wrap with DataParallel if it was wrapped before
            if was_dataparallel:
                model = DataParallel(model)
                model.to("cuda")
            else:
                model.to(self.model_device_num)
        else:
            model = base_model
        
        if frozen:
            self.freeze_model(model)
        
        trainable_params = self.count_trainable_parameters(model)
        total_params = self.count_parameters(model)
        logging.info(
            f"Model has {trainable_params:,} trainable parameters, "
            f"{total_params:,} total parameters."
        )
        
        # Create optimizer - only update student parameters if using semi-supervised
        if use_semi_supervised:
            # Get student parameters (unwrap DataParallel if needed)
            if isinstance(model, DataParallel):
                student_params = model.module.student.parameters()
            else:
                student_params = model.student.parameters()
            
            if self.use_sam:
                base_optimizer = torch.optim.AdamW
                optimizer = SAM(
                    student_params,
                    base_optimizer,
                    lr=learning_rate,
                    adaptive=self.adaptive_sam
                )
            else:
                optimizer = self.create_optimizer(model, learning_rate, params=student_params)
        else:
            if self.use_sam:
                base_optimizer = torch.optim.AdamW
                optimizer = SAM(
                    model.parameters(),
                    base_optimizer,
                    lr=learning_rate,
                    adaptive=self.adaptive_sam
                )
            else:
                optimizer = self.create_optimizer(model, learning_rate)
        
        logging.info("Model and optimizer created.")
        return model, optimizer
    
    def load_encoder_weights(
        self,
        model: nn.Module,
        weights_fname: Path,
        gpu: bool = True,
        device_num: int = 0
    ):
        """
        Load encoder weights from a saved checkpoint.
        
        Args:
            model: Model to load weights into
            weights_fname: Path to weights file
            gpu: Whether to load on GPU
            device_num: Device number
        """
        map_location = f"cuda:{device_num}" if gpu else "cpu"
        weights_fname = weights_fname.resolve()
        
        resnet_model = models.resnet50(pretrained=True)
        logging.info(f"Loading saved weights from: {weights_fname}")
        
        checkpoint = torch.load(weights_fname, map_location=map_location, weights_only=False)
        resnet_model.load_state_dict(checkpoint, strict=False)
        
        # Adapt first conv layer for single-channel input
        new_in_channels = 1
        default_in_channels = 3
        for module in resnet_model.modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
                break
        weight = module.weight.detach()
        module.in_channels = new_in_channels
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)
        
        # Unwrap DataParallel if needed
        if isinstance(model, DataParallel):
            model.module.encoder.load_state_dict(resnet_model.state_dict(), strict=False)
        else:
            model.encoder.load_state_dict(resnet_model.state_dict(), strict=False)
        model.to(map_location)
    
    def freeze_model(self, model: nn.Module):
        """Freeze all encoder parameters (not just conv layers)."""
        trainable_before = self.count_trainable_parameters(model)
        logging.info(
            f"Freezing encoder layers. Before: {trainable_before:,} trainable."
        )
        frozen_count = 0
        
        # Unwrap DataParallel if needed
        model_to_check = model.module if isinstance(model, DataParallel) else model
        
        for name, param in model_to_check.named_parameters():
            if "encoder" in name and param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
        
        trainable_after = self.count_trainable_parameters(model)
        logging.info(
            f"After freezing: {trainable_after:,} trainable. "
            f"Froze {frozen_count} encoder parameter groups."
        )
    
    def unfreeze_model(self, model: nn.Module):
        """Unfreeze all encoder parameters."""
        logging.info(f"Unfreezing encoder layers.")
        unfrozen_count = 0
        
        # Unwrap DataParallel if needed
        model_to_check = model.module if isinstance(model, DataParallel) else model
        
        for name, param in model_to_check.named_parameters():
            if "encoder" in name and not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
        
        trainable_after = self.count_trainable_parameters(model)
        logging.info(
            f"After unfreezing: {trainable_after:,} trainable. "
            f"Unfroze {unfrozen_count} encoder parameter groups."
        )
    
    def count_trainable_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def count_parameters(self, model: nn.Module) -> int:
        """Count total parameters in model."""
        return sum(p.numel() for p in model.parameters())
    
    def create_optimizer(
        self,
        model: nn.Module,
        learning_rate: float,
        params=None
    ):
        """
        Create optimizer for model.
        
        Args:
            model: Model to create optimizer for
            learning_rate: Learning rate
            params: Optional parameter group (if None, uses model.parameters())
        
        Returns:
            Optimizer
        """
        if params is None:
            params = model.parameters()
        return torch.optim.AdamW(params, lr=learning_rate)
    
    def create_onecycle_lr_scheduler(
        self,
        optimizer,
        num_epochs: int,
        lr_to_use: float,
        steps_per_epoch: int,
        pct_start: float = 0.3,
    ):
        """
        Create OneCycleLR scheduler.
        
        Args:
            optimizer: Optimizer
            num_epochs: Number of epochs
            lr_to_use: Maximum learning rate
            steps_per_epoch: Steps per epoch
            pct_start: Percentage of cycle for warm-up
        
        Returns:
            OneCycleLR scheduler
        """
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr_to_use,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=pct_start,
        )
    
    def create_early_stopping(
        self,
        output_path: Path,
        patience: int,
        model_struc_dict: dict,
        best_score: float = None,
    ):
        """
        Create early stopping callback.
        
        Args:
            output_path: Path to save checkpoints
            patience: Patience for early stopping
            model_struc_dict: Model structure dictionary
            best_score: Best score to start with
        
        Returns:
            EarlyStopping instance
        """
        return EarlyStopping(
            patience=patience,
            verbose=True,
            path=output_path,
            model_dict=model_struc_dict,
            best_score=best_score,
        )
    
    def load_model_weights(
        self,
        model: nn.Module,
        optimizer,
        checkpoint_path: Path,
        gpu: bool = True,
        load_optimizer: bool = False,
        use_semi_supervised: bool = None,
    ) -> float:
        """
        Load model and optionally optimizer weights from checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state into (if load_optimizer=True)
            checkpoint_path: Path to checkpoint
            gpu: Whether to load on GPU
            load_optimizer: Whether to load optimizer state
            use_semi_supervised: Whether model uses semi-supervised learning
        
        Returns:
            Validation loss from checkpoint
        """
        if use_semi_supervised is None:
            use_semi_supervised = self.use_semi_supervised
        
        map_location = f"cuda:{self.model_device_num}" if gpu else "cpu"
        model_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        logging.info("Loading model weights.")
        
        # Handle MeanTeacherModel state dict
        if use_semi_supervised:
            # Unwrap DataParallel if needed
            if isinstance(model, DataParallel):
                model.module.load_state_dict(model_dict["model_state_dict"])
                # Restore glob_it if present
                if 'glob_it' in model_dict:
                    model.module.glob_it = model_dict['glob_it']
            else:
                model.load_state_dict(model_dict["model_state_dict"])
                # Restore glob_it if present
                if 'glob_it' in model_dict:
                    model.glob_it = model_dict['glob_it']
        else:
            model.load_state_dict(model_dict["model_state_dict"])
        
        if load_optimizer:
            logging.info("Loading optimizer weights.")
            optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        
        return model_dict.get("loss_val", np.inf)
