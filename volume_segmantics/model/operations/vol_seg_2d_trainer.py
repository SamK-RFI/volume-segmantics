import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Union, Dict, List, Optional, Tuple
import copy

import matplotlib as mpl
mpl.use("Agg")

import numpy as np
import termplotlib as tpl

import torch
import torch.nn as nn
from torchvision import models
from torch import nn
import torch.nn.functional as F

import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from matplotlib import pyplot as plt
from tqdm import tqdm
from volume_segmantics.data.dataloaders import get_2d_training_dataloaders
from volume_segmantics.data.pytorch3dunet_losses import (
    BCEDiceLoss,
    DiceLoss,
    GeneralizedDiceLoss,
    BoundaryDoULoss,
    BoundaryDoULossV2,
    TverskyLoss,
    BoundaryLoss,
    BoundaryDoUDiceLoss
)
from volume_segmantics.data.pytorch3dunet_metrics import (
    DiceCoefficient,
    MeanIoU,
)
from volume_segmantics.model.model_2d import create_model_on_device, create_model_from_file_full_weights
from volume_segmantics.utilities.early_stopping import EarlyStopping
from volume_segmantics.model.sam import SAM



class ClassWeightedDiceLoss(nn.Module):
    """
    Dice loss with per-class weighting for multi-class segmentation.
    
    Supports multiple weighting strategies:
    - 'uniform': Equal weight for all classes
    - 'inverse_freq': Weight inversely proportional to class frequency
    - 'inverse_sqrt_freq': Weight inversely proportional to sqrt of frequency
    - 'custom': User-provided weights
    
    Args:
        num_classes: Number of segmentation classes
        weight_mode: Weighting strategy ('uniform', 'inverse_freq', 'inverse_sqrt_freq', 'custom')
        class_weights: Custom weights when weight_mode='custom' (list or tensor)
        exclude_background: Whether to exclude class 0 from loss computation
        smooth: Smoothing factor to avoid division by zero
        softmax: Whether to apply softmax to predictions (set False if already applied)
    """
    
    def __init__(
        self,
        num_classes: int,
        weight_mode: str = "uniform",
        class_weights: Optional[List[float]] = None,
        exclude_background: bool = False,
        smooth: float = 1e-7,
        softmax: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_mode = weight_mode
        self.exclude_background = exclude_background
        self.smooth = smooth
        self.softmax = softmax
        
        # Initialize weights
        if weight_mode == "custom" and class_weights is not None:
            self.register_buffer(
                "class_weights", 
                torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "class_weights",
                torch.ones(num_classes, dtype=torch.float32)
            )
        
        # Track class frequencies for adaptive weighting
        self.register_buffer("class_pixel_counts", torch.zeros(num_classes))
        self.register_buffer("total_pixels", torch.tensor(0.0))
        
        logging.info(
            f"ClassWeightedDiceLoss initialized: num_classes={num_classes}, "
            f"weight_mode={weight_mode}, exclude_background={exclude_background}"
        )
    
    def _compute_weights_from_batch(
        self, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute class weights from the current batch based on frequency."""
        # Convert MetaTensor to regular tensor if needed (MONAI)
        if hasattr(target, 'as_tensor'):
            target = target.as_tensor()
        
        # target shape: (B, C, H, W) one-hot or (B, H, W) class indices
        if target.dim() == 4:
            # One-hot encoded
            class_counts = target.sum(dim=(0, 2, 3))  # (C,)
            # Convert to regular tensor if MetaTensor
            if hasattr(class_counts, 'as_tensor'):
                class_counts = class_counts.as_tensor()
        else:
            class_counts = torch.zeros(
                self.num_classes, device=target.device, dtype=torch.float32
            )
            for c in range(self.num_classes):
                count = (target == c).sum()
                if hasattr(count, 'as_tensor'):
                    count = count.as_tensor()
                class_counts[c] = count
        
        total = class_counts.sum()

        if hasattr(total, 'as_tensor'):
            total = total.as_tensor()
        
        if self.weight_mode == "inverse_freq":
            freq = class_counts / (total + self.smooth)
            weights = 1.0 / (freq + self.smooth)
        elif self.weight_mode == "inverse_sqrt_freq":
            freq = class_counts / (total + self.smooth)
            weights = 1.0 / (torch.sqrt(freq) + self.smooth)
        else:
            # uniform 
            weights = torch.ones_like(class_counts)
        
        # Normalize weights to sum to num_classes
        weight_sum = weights.sum()

        if hasattr(weight_sum, 'as_tensor'):
            weight_sum = weight_sum.as_tensor()
        weights = weights * self.num_classes / (weight_sum + self.smooth)
        
        # Ensure return value is a regular tensor
        if hasattr(weights, 'as_tensor'):
            weights = weights.as_tensor()
        
        return weights
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class-weighted Dice loss.
        
        Args:
            pred: Predictions of shape (B, C, H, W) - logits or probabilities
            target: Targets of shape (B, C, H, W) one-hot or (B, H, W) indices
        
        Returns:
            Scalar loss value
        """
        if hasattr(target, 'as_tensor'):
            target = target.as_tensor()
        if hasattr(pred, 'as_tensor'):
            pred = pred.as_tensor()
        
        if self.softmax:
            pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot if needed
        if target.dim() == 3:
            # (B, H, W) -> (B, C, H, W)
            target = F.one_hot(target.long(), self.num_classes)
            target = target.permute(0, 3, 1, 2).float()
        
        if self.weight_mode in ("inverse_freq", "inverse_sqrt_freq"):
            weights = self._compute_weights_from_batch(target)
            if hasattr(weights, 'as_tensor'):
                weights = weights.as_tensor()
        else:
            weights = self.class_weights
        
        if hasattr(weights, 'as_tensor'):
            weights = weights.as_tensor()
        elif not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=target.device, dtype=target.dtype)
        
        # Compute per-class Dice
        # (B, C, H, W) -> (B, C, H*W)
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
        
        # Intersection and union per class per sample
        intersection = (pred_flat * target_flat).sum(dim=2)  # (B, C)
        pred_sum = pred_flat.sum(dim=2)  # (B, C)
        target_sum = target_flat.sum(dim=2)  # (B, C)
        
        # Dice per class per sample
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        start_class = 1 if self.exclude_background else 0
        
        if self.exclude_background:
            # Zero out background weight
            weights = weights.clone()
            weights[0] = 0.0
        
        # Weighted average across classes, then average across batch
        # weights: (C,) -> (1, C) for broadcasting
        weighted_dice = dice * weights.unsqueeze(0)  # (B, C)
        
        # Sum weighted dice and normalize by sum of weights
        weight_sum = weights[start_class:].sum()
        if hasattr(weight_sum, 'as_tensor'):
            weight_sum = weight_sum.as_tensor()
        loss = 1.0 - weighted_dice[:, start_class:].sum(dim=1) / (weight_sum + self.smooth)
        
        return loss.mean()
    
    def get_current_weights(self) -> torch.Tensor:
        """Return current class weights for logging."""
        return self.class_weights.clone()


class CombinedCEDiceLoss(nn.Module):
    """
    Combined Cross-Entropy and class-weighted Dice loss.
    
    Loss = alpha * CE + beta * Dice
    
    Args:
        num_classes: Number of segmentation classes
        alpha: Weight for Cross-Entropy loss
        beta: Weight for Dice loss
        dice_weight_mode: Weighting strategy for Dice loss
        class_weights_ce: Optional class weights for CE loss
        exclude_background: Whether to exclude background from Dice
    """
    
    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        dice_weight_mode: str = "inverse_sqrt_freq",
        class_weights_ce: Optional[List[float]] = None,
        exclude_background: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        # Cross-Entropy loss
        if class_weights_ce is not None:
            ce_weights = torch.tensor(class_weights_ce, dtype=torch.float32)
            self.ce_loss = nn.CrossEntropyLoss(weight=ce_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # Dice loss
        self.dice_loss = ClassWeightedDiceLoss(
            num_classes=num_classes,
            weight_mode=dice_weight_mode,
            exclude_background=exclude_background,
            softmax=True,
        )
        
        logging.info(
            f"CombinedCEDiceLoss: alpha={alpha}, beta={beta}, "
            f"dice_weight_mode={dice_weight_mode}"
        )
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, C, H, W) one-hot or (B, H, W) class indices
        """
        # Prepare target for CE (needs class indices)
        if target.dim() == 4:
            target_ce = torch.argmax(target, dim=1)
        else:
            target_ce = target
        
        ce = self.ce_loss(pred, target_ce)
        dice = self.dice_loss(pred, target)
        
        return self.alpha * ce + self.beta * dice


@dataclass
class MultiTaskLossTracker:
    """Tracks individual task losses across batches for reporting."""
    seg_losses: List[float] = field(default_factory=list)
    boundary_losses: List[float] = field(default_factory=list)
    task3_losses: List[float] = field(default_factory=list)
    total_losses: List[float] = field(default_factory=list)
    
    # For eval metrics per task
    seg_dice_scores: List[float] = field(default_factory=list)
    boundary_dice_scores: List[float] = field(default_factory=list)
    
    # Per-class Dice scores
    per_class_dice: Dict[int, List[float]] = field(default_factory=dict)
    
    def append_losses(self, losses: Dict[str, float]):
        """Append loss values from a batch."""
        if "seg" in losses:
            self.seg_losses.append(losses["seg"])
        if "boundary" in losses:
            self.boundary_losses.append(losses["boundary"])
        if "task3" in losses:
            self.task3_losses.append(losses["task3"])
        if "total" in losses:
            self.total_losses.append(losses["total"])
    
    def append_metrics(self, metrics: Dict[str, float]):
        """Append evaluation metrics from a batch."""
        if "seg_dice" in metrics:
            self.seg_dice_scores.append(metrics["seg_dice"])
        if "boundary_dice" in metrics:
            self.boundary_dice_scores.append(metrics["boundary_dice"])
        
        # Per-class Dice
        for key, value in metrics.items():
            if key.startswith("dice_class_"):
                class_idx = int(key.split("_")[-1])
                if class_idx not in self.per_class_dice:
                    self.per_class_dice[class_idx] = []
                if not math.isnan(value):
                    self.per_class_dice[class_idx].append(value)
    
    def get_average_losses(self) -> Dict[str, float]:
        """Get average losses across all tracked batches."""
        result = {}
        if self.seg_losses:
            result["seg"] = float(np.average(self.seg_losses))
        if self.boundary_losses:
            result["boundary"] = float(np.average(self.boundary_losses))
        if self.task3_losses:
            result["task3"] = float(np.average(self.task3_losses))
        if self.total_losses:
            result["total"] = float(np.average(self.total_losses))
        return result
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all tracked batches."""
        result = {}
        if self.seg_dice_scores:
            result["seg_dice"] = float(np.average(self.seg_dice_scores))
        if self.boundary_dice_scores:
            result["boundary_dice"] = float(np.average(self.boundary_dice_scores))
        
        # Per-class averages
        for class_idx, scores in self.per_class_dice.items():
            if scores:
                result[f"dice_class_{class_idx}"] = float(np.average(scores))
        
        return result
    
    def clear(self):
        """Clear all tracked values for next epoch."""
        self.seg_losses.clear()
        self.boundary_losses.clear()
        self.task3_losses.clear()
        self.total_losses.clear()
        self.seg_dice_scores.clear()
        self.boundary_dice_scores.clear()
        self.per_class_dice.clear()


class MultiTaskLossCalculator:
    """
    Handles multi-task loss calculation with configurable weights and loss functions.
    
    Supports:
    - Segmentation (primary task): CE or Dice-based losses
    - Boundary detection (auxiliary): BCE or Dice
    - Optional task3: BCE or custom
    """
    
    def __init__(
        self,
        seg_criterion: nn.Module,
        seg_weight: float = 1.0,
        boundary_weight: float = 1.0,
        task3_weight: float = 1.0,
        use_cross_entropy: bool = False,
        boundary_loss_type: str = "bce",  # "bce", "dice", or "bce_dice"
        num_classes: int = 6,
    ):
        self.seg_criterion = seg_criterion
        self.seg_weight = seg_weight
        self.boundary_weight = boundary_weight
        self.task3_weight = task3_weight
        self.use_cross_entropy = use_cross_entropy
        self.boundary_loss_type = boundary_loss_type
        self.num_classes = num_classes
        
        if boundary_loss_type in ("dice", "bce_dice"):
            self.boundary_dice = DiceLoss(normalization="sigmoid")
    
    def _compute_boundary_loss(
        self, 
        output: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute boundary loss with configured loss type."""
        if self.boundary_loss_type == "bce":
            return F.binary_cross_entropy_with_logits(output, target)
        elif self.boundary_loss_type == "dice":
            return self.boundary_dice(output, target)
        elif self.boundary_loss_type == "bce_dice":
            bce = F.binary_cross_entropy_with_logits(output, target)
            dice = self.boundary_dice(output, target)
            return 0.5 * bce + 0.5 * dice
        else:
            raise ValueError(f"Unknown boundary loss type: {self.boundary_loss_type}")
    
    def compute(
        self,
        outputs: Tuple[torch.Tensor, ...],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute individual and total losses for all tasks.
        
        Args:
            outputs: Tuple of model outputs (seg_output, boundary_output, ...)
            targets: Dict with task targets {"seg": tensor, "boundary": tensor, ...}
        
        Returns:
            Dict with individual task losses and weighted 'total' loss
        """
        losses = {}
        total = torch.tensor(0.0, device=outputs[0].device)
        
        # Segmentation Loss (Task 1) 
        if "seg" in targets:
            seg_output = outputs[0]
            seg_target = targets["seg"]
            
            if self.use_cross_entropy:
                # CrossEntropyLoss expects class indices, not one-hot
                seg_loss = self.seg_criterion(
                    seg_output, 
                    torch.argmax(seg_target, dim=1)
                )
            else:
                seg_loss = self.seg_criterion(seg_output, seg_target.float())
            
            losses["seg"] = seg_loss
            total = total + self.seg_weight * seg_loss
        
        # Boundary Loss (Task 2) 
        if "boundary" in targets:
            boundary_target = targets["boundary"]
            
            if len(outputs) < 2:
                raise ValueError(
                    "Model has only 1 output but boundary target provided. "
                    "Check MultitaskUnet configuration (num_tasks >= 2)."
                )
            
            boundary_output = outputs[1]
            
            # Validate and handle channel mismatch
            out_channels = boundary_output.shape[1]
            target_channels = boundary_target.shape[1]
            
            if out_channels != target_channels:
                if out_channels > target_channels:
                    logging.warning(
                        f"Boundary output has {out_channels} channels but target has "
                        f"{target_channels}. Slicing output to match. Consider fixing "
                        "decoder output channels in model config."
                    )
                    boundary_output = boundary_output[:, :target_channels, :, :]
                else:
                    raise ValueError(
                        f"Boundary output channels ({out_channels}) < target channels "
                        f"({target_channels}). Model misconfiguration."
                    )
            
            boundary_loss = self._compute_boundary_loss(boundary_output, boundary_target)
            losses["boundary"] = boundary_loss
            total = total + self.boundary_weight * boundary_loss
        
        # Task 3 Loss
        if "task3" in targets:
            task3_target = targets["task3"]
            
            if len(outputs) < 3:
                raise ValueError(
                    "Model has < 3 outputs but task3 target provided. "
                    "Check MultitaskUnet configuration (num_tasks >= 3)."
                )
            
            task3_output = outputs[2]
            
            # Handle channel mismatch
            if task3_output.shape[1] != task3_target.shape[1]:
                task3_output = task3_output[:, :task3_target.shape[1], :, :]
            
            task3_loss = F.binary_cross_entropy_with_logits(task3_output, task3_target)
            losses["task3"] = task3_loss
            total = total + self.task3_weight * task3_loss
        
        losses["total"] = total
        return losses



class VolSeg2dTrainer:
    """
    Class that provides methods to train a 2d deep learning model
    with support for multi-task learning and detailed loss tracking.
    """

    def __init__(
        self,
        image_dir_path: Path,
        label_dir_path: Path,
        labels: Union[int, dict],
        settings: SimpleNamespace,
    ):
        """
        Inits VolSeg2dTrainer.

        Args:
            image_dir_path: Path to directory containing image data slices.
            label_dir_path: Path to directory containing label data slices.
            labels: Either number of labels or dictionary containing label names.
            settings: A training settings object.
        """
        self.training_loader, self.validation_loader = get_2d_training_dataloaders(
            image_dir_path, label_dir_path, settings
        )
        self.label_no = labels if isinstance(labels, int) else len(labels)
        self.codes = labels if isinstance(labels, dict) else {}
        self.settings = settings
        
        # Learning rate finder params
        self.starting_lr = float(settings.starting_lr)
        self.end_lr = float(settings.end_lr)
        self.log_lr_ratio = self._calculate_log_lr_ratio()
        self.lr_find_epochs = settings.lr_find_epochs
        self.lr_reduce_factor = settings.lr_reduce_factor
        
        # Params for model training
        self.model_device_num = int(settings.cuda_device)
        self.patience = settings.patience
        self.loss_criterion = self._get_loss_criterion()
        self.eval_metric = self._get_eval_metric()
        self.model_struc_dict = self._get_model_struc_dict(settings)
        
        # Multi-task configuration
        self.use_multitask = getattr(settings, "use_multitask", False)
        if self.use_multitask:
            self._setup_multitask_loss_calculator(settings)
        
        # Dice evaluation settings
        self.exclude_background_from_dice = getattr(
            settings, "exclude_background_from_dice", True
        )
        self.dice_averaging = getattr(settings, "dice_averaging", "macro")
        
        # Loss tracking 
        self.train_loss_tracker = MultiTaskLossTracker()
        self.valid_loss_tracker = MultiTaskLossTracker()
        
        # Epoch history for plots
        self.epoch_history = {
            "train_total": [],
            "train_seg": [],
            "train_boundary": [],
            "train_task3": [],
            "valid_total": [],
            "valid_seg": [],
            "valid_boundary": [],
            "valid_task3": [],
            "seg_dice": [],
            "boundary_dice": [],
        }
        
        # Per-class Dice history
        for c in range(self.label_no):
            self.epoch_history[f"dice_class_{c}"] = []
        
        self.avg_train_losses = self.epoch_history["train_total"]
        self.avg_valid_losses = self.epoch_history["valid_total"]
        self.avg_eval_scores = self.epoch_history["seg_dice"]
        
        if str(settings.encoder_weights_path) != "False":
            self.encoder_weights_path = Path(settings.encoder_weights_path)
        else:
            self.encoder_weights_path = False
        if str(settings.full_weights_path) != "False":
            self.full_weights_path = Path(settings.full_weights_path)
        else:
            self.full_weights_path = False
            
        # Sharpness Aware Minimisation optimizer settings
        self.use_sam = settings.use_sam
        self.adaptive_sam = settings.adaptive_sam
    
    def _setup_multitask_loss_calculator(self, settings):
        """Initialize the multi-task loss calculator with settings."""
        self.loss_calculator = MultiTaskLossCalculator(
            seg_criterion=self.loss_criterion,
            seg_weight=getattr(settings, "seg_loss_weight", 1.0),
            boundary_weight=getattr(settings, "boundary_loss_weight", 1.0),
            task3_weight=getattr(settings, "task3_loss_weight", 1.0),
            use_cross_entropy=(settings.loss_criterion == "CrossEntropyLoss"),
            boundary_loss_type=getattr(settings, "boundary_loss_type", "bce"),
            num_classes=self.label_no,
        )
        logging.info(
            f"Multi-task loss calculator initialized: "
            f"seg_weight={self.loss_calculator.seg_weight}, "
            f"boundary_weight={self.loss_calculator.boundary_weight}, "
            f"boundary_loss_type={self.loss_calculator.boundary_loss_type}"
        )

    def _get_model_struc_dict(self, settings):
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

    def _calculate_log_lr_ratio(self):
        return math.log(self.end_lr / self.starting_lr)

    def _create_model_and_optimiser(self, learning_rate, frozen=False):
        logging.info(f"Setting up the model on device {self.settings.cuda_device}.")
        self.model = create_model_on_device(
            self.model_device_num, self.model_struc_dict
        )
        if frozen:
            self._freeze_model()
        logging.info(
            f"Model has {self._count_trainable_parameters():,} trainable parameters, "
            f"{self._count_parameters():,} total parameters."
        )
        
        if self.use_sam:
            base_optimizer = torch.optim.AdamW
            self.optimizer = SAM(
                self.model.parameters(), 
                base_optimizer, 
                lr=learning_rate, 
                adaptive=self.adaptive_sam
            )
        else:
            self.optimizer = self._create_optimizer(learning_rate)
        
        self._log_model_architecture_details()
        self._log_learning_rates()

        logging.info("Trainer created.")

    def _load_encoder_weights(self, weights_fname: Path, gpu: bool = True, device_num: int = 0):
        map_location = f"cuda:{device_num}" if gpu else "cpu"
        weights_fname = weights_fname.resolve()
        
        model = models.resnet50(pretrained=True)
        logging.info(f"Loading saved weights from: {weights_fname}")
           
        checkpoint = torch.load(weights_fname, map_location=map_location, weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        
        # Adapt first conv layer for single-channel input
        new_in_channels = 1
        default_in_channels = 3    
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
                break
        weight = module.weight.detach()
        module.in_channels = new_in_channels
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)
        
        self.model.encoder.load_state_dict(model.state_dict(), strict=False)
        self.model.to(map_location)

    def _freeze_model(self):
        """Freeze all encoder parameters (not just conv layers)."""
        logging.info(
            f"Freezing encoder layers. Before: {self._count_trainable_parameters():,} trainable."
        )
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if "encoder" in name and param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
        
        logging.info(
            f"After freezing: {self._count_trainable_parameters():,} trainable. "
            f"Froze {frozen_count} encoder parameter groups."
        )

    def _unfreeze_model(self):
        """Unfreeze all encoder parameters."""
        logging.info(f"Unfreezing encoder layers.")
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if "encoder" in name and not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
        logging.info(
            f"After unfreezing: {self._count_trainable_parameters():,} trainable. "
            f"Unfroze {unfrozen_count} encoder parameter groups."
        )

    def _count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
    
    def _log_model_architecture_details(self):
        """Log detailed information about model architecture and trainability."""
        logging.info("=" * 80)
        logging.info("MODEL ARCHITECTURE DIAGNOSTICS")
        logging.info("=" * 80)
        
        total_params = self._count_parameters()
        trainable_params = self._count_trainable_parameters()
        frozen_params = total_params - trainable_params
        
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        logging.info(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        
        component_stats = {
            "encoder": {"trainable": 0, "frozen": 0, "total": 0},
            "decoder": {"trainable": 0, "frozen": 0, "total": 0},
            "head": {"trainable": 0, "frozen": 0, "total": 0},
            "other": {"trainable": 0, "frozen": 0, "total": 0},
        }
        
        head_trainability = {}
        
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            is_trainable = param.requires_grad
            
            if "encoder" in name:
                component = "encoder"
            elif "decoder" in name or "decoders" in name:
                component = "decoder"
            elif "head" in name or "heads" in name:
                component = "head"
                head_idx = None
                for i in range(len(self.model.heads) if hasattr(self.model, 'heads') else 0):
                    if f"heads.{i}" in name or f"head.{i}" in name:
                        head_idx = i
                        break
                if head_idx is not None:
                    if head_idx not in head_trainability:
                        head_trainability[head_idx] = {"trainable": 0, "frozen": 0, "total": 0}
                    if is_trainable:
                        head_trainability[head_idx]["trainable"] += num_params
                    else:
                        head_trainability[head_idx]["frozen"] += num_params
                    head_trainability[head_idx]["total"] += num_params
            else:
                component = "other"
            
            component_stats[component]["total"] += num_params
            if is_trainable:
                component_stats[component]["trainable"] += num_params
            else:
                component_stats[component]["frozen"] += num_params
        
        logging.info("\nComponent Breakdown:")
        for component, stats in component_stats.items():
            if stats["total"] > 0:
                pct_trainable = 100 * stats["trainable"] / stats["total"]
                logging.info(
                    f"  {component.capitalize()}: "
                    f"{stats['trainable']:,} trainable, {stats['frozen']:,} frozen, "
                    f"{stats['total']:,} total ({pct_trainable:.2f}% trainable)"
                )
        
        if hasattr(self.model, 'heads') and len(self.model.heads) > 1:
            logging.info("\nHead Trainability (Multitask Model):")
            for head_idx, stats in sorted(head_trainability.items()):
                pct_trainable = 100 * stats["trainable"] / stats["total"] if stats["total"] > 0 else 0
                head_name = f"Head {head_idx}"
                if head_idx == 0:
                    head_name += " (Segmentation)"
                elif head_idx == 1:
                    head_name += " (Boundary)"
                elif head_idx == 2:
                    head_name += " (Task3)"
                
                logging.info(
                    f"  {head_name}: "
                    f"{stats['trainable']:,} trainable, {stats['frozen']:,} frozen, "
                    f"{stats['total']:,} total ({pct_trainable:.2f}% trainable)"
                )
                
                if stats["trainable"] == 0:
                    logging.warning(f"  ⚠️  WARNING: {head_name} has NO trainable parameters!")
        
        if self.use_multitask and hasattr(self.model, 'heads') and len(self.model.heads) > 1:
            boundary_head_idx = 1
            if boundary_head_idx in head_trainability:
                boundary_stats = head_trainability[boundary_head_idx]
                if boundary_stats["trainable"] == 0:
                    logging.error("  ❌ CRITICAL: Boundary head has NO trainable parameters!")
                else:
                    logging.info(f"  ✓ Boundary head is trainable ({boundary_stats['trainable']:,} params)")
        
        logging.info("=" * 80)
    
    def _log_gradient_statistics(self, epoch: int, batch_idx: int = None):
        """Log gradient statistics to diagnose training issues."""
        if not hasattr(self, '_grad_log_counter'):
            self._grad_log_counter = 0
        
        self._grad_log_counter += 1
        
        if self._grad_log_counter % 50 != 0 and batch_idx is not None:
            return
        
        grad_stats = {
            "encoder": {"mean": [], "max": [], "min": [], "count": 0, "zero_count": 0},
            "decoder": {"mean": [], "max": [], "min": [], "count": 0, "zero_count": 0},
            "head": {"mean": [], "max": [], "min": [], "count": 0, "zero_count": 0},
        }
        
        head_grads = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad = param.grad.data
                grad_mean = grad.abs().mean().item()
                grad_max = grad.abs().max().item()
                grad_min = grad.abs().min().item()
                zero_grads = (grad == 0).sum().item()
                total_grads = grad.numel()
                
                if "encoder" in name:
                    component = "encoder"
                elif "decoder" in name or "decoders" in name:
                    component = "decoder"
                elif "head" in name or "heads" in name:
                    component = "head"
                    head_idx = None
                    for i in range(len(self.model.heads) if hasattr(self.model, 'heads') else 0):
                        if f"heads.{i}" in name or f"head.{i}" in name:
                            head_idx = i
                            break
                    if head_idx is not None:
                        if head_idx not in head_grads:
                            head_grads[head_idx] = {"mean": [], "max": [], "zero_ratio": []}
                        head_grads[head_idx]["mean"].append(grad_mean)
                        head_grads[head_idx]["max"].append(grad_max)
                        head_grads[head_idx]["zero_ratio"].append(zero_grads / total_grads)
                else:
                    continue
                
                grad_stats[component]["mean"].append(grad_mean)
                grad_stats[component]["max"].append(grad_max)
                grad_stats[component]["min"].append(grad_min)
                grad_stats[component]["count"] += 1
                grad_stats[component]["zero_count"] += zero_grads
            elif param.requires_grad and param.grad is None:
                if "head" in name or "heads" in name:
                    head_idx = None
                    for i in range(len(self.model.heads) if hasattr(self.model, 'heads') else 0):
                        if f"heads.{i}" in name or f"head.{i}" in name:
                            head_idx = i
                            break
                    if head_idx == 1:
                        logging.warning(f"Boundary head parameter '{name}' has no gradient!")
        
        log_prefix = f"Epoch {epoch}"
        if batch_idx is not None:
            log_prefix += f", Batch {batch_idx}"
        
        logging.info(f"\n{log_prefix} - Gradient Statistics:")
        for component, stats in grad_stats.items():
            if stats["count"] > 0:
                mean_grad = np.mean(stats["mean"]) if stats["mean"] else 0
                max_grad = np.max(stats["max"]) if stats["max"] else 0
                min_grad = np.min(stats["min"]) if stats["min"] else 0
                zero_ratio = stats["zero_count"] / sum(p.numel() for n, p in self.model.named_parameters() 
                                                      if component in n and p.requires_grad) if stats["count"] > 0 else 0
                
                logging.info(
                    f"  {component.capitalize()}: "
                    f"mean={mean_grad:.2e}, max={max_grad:.2e}, min={min_grad:.2e}, "
                    f"zero_ratio={zero_ratio:.4f}"
                )
        
        if head_grads:
            logging.info("  Head-specific gradients:")
            for head_idx, stats in sorted(head_grads.items()):
                head_name = f"Head {head_idx}"
                if head_idx == 0:
                    head_name += " (Seg)"
                elif head_idx == 1:
                    head_name += " (Boundary)"
                elif head_idx == 2:
                    head_name += " (Task3)"
                
                mean_grad = np.mean(stats["mean"]) if stats["mean"] else 0
                max_grad = np.max(stats["max"]) if stats["max"] else 0
                zero_ratio = np.mean(stats["zero_ratio"]) if stats["zero_ratio"] else 0
                
                logging.info(
                    f"    {head_name}: mean={mean_grad:.2e}, max={max_grad:.2e}, "
                    f"zero_ratio={zero_ratio:.4f}"
                )
                
                if head_idx == 1 and mean_grad < 1e-7:
                    logging.warning(f"Boundary head has very small gradients.")
    
    def _log_learning_rates(self):
        """Log current learning rates for different parameter groups."""
        if hasattr(self.optimizer, 'param_groups'):
            logging.info("Learning Rates:")
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group.get('lr', 'N/A')
                num_params = sum(p.numel() for p in param_group['params'] if p.requires_grad)
                logging.info(f"  Group {i}: lr={lr}, params={num_params:,}")
        else:
            if hasattr(self.optimizer, 'base'):
                base_lr = self.optimizer.base.param_groups[0].get('lr', 'N/A')
                logging.info(f"Learning Rate (SAM): {base_lr}")
    
    def _log_parameter_statistics(self, epoch: int):
        """Log parameter value statistics to track if parameters are changing."""
        if not hasattr(self, '_param_stats_history'):
            self._param_stats_history = {}
        
        current_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                param_min = param.data.min().item()
                param_max = param.data.max().item()
                
                component = None
                if "encoder" in name:
                    component = "encoder"
                elif "decoder" in name or "decoders" in name:
                    component = "decoder"
                elif "head" in name or "heads" in name:
                    component = "head"
                    if "heads.1" in name or "head.1" in name:
                        if "boundary" not in current_stats:
                            current_stats["boundary"] = []
                        current_stats["boundary"].append({
                            "mean": param_mean,
                            "std": param_std,
                            "min": param_min,
                            "max": param_max,
                        })
                
                if component:
                    if component not in current_stats:
                        current_stats[component] = []
                    current_stats[component].append({
                        "mean": param_mean,
                        "std": param_std,
                        "min": param_min,
                        "max": param_max,
                    })
        
        if epoch == 1 or epoch % 5 == 0:
            logging.info(f"\nEpoch {epoch} - Parameter Statistics:")
            for component, stats_list in current_stats.items():
                if stats_list:
                    mean_vals = [s["mean"] for s in stats_list]
                    std_vals = [s["std"] for s in stats_list]
                    
                    logging.info(
                        f"  {component.capitalize()}: "
                        f"mean={np.mean(mean_vals):.6f}, std={np.mean(std_vals):.6f}"
                    )
                    
                    if component == "boundary" and epoch > 1:
                        if "boundary" in self._param_stats_history:
                            prev_mean = np.mean([s["mean"] for s in self._param_stats_history["boundary"]])
                            curr_mean = np.mean(mean_vals)
                            change = abs(curr_mean - prev_mean)
                            if change < 1e-6:
                                logging.warning(f"Boundary head parameters changed by only {change:.2e}")
        
        self._param_stats_history = current_stats
    
    def _log_boundary_prediction_statistics(self, epoch: int):
        """Log detailed boundary prediction statistics to diagnose Dice issues."""
        if not self._boundary_stats:
            return
        
        avg_dice = np.mean([s["dice"] for s in self._boundary_stats])
        avg_dice_prob = np.mean([s.get("dice_prob", 0) for s in self._boundary_stats])
        avg_threshold = np.mean([s.get("threshold", 0.5) for s in self._boundary_stats])
        avg_mean_prob = np.mean([s["mean_prob"] for s in self._boundary_stats])
        avg_max_prob = np.mean([s["max_prob"] for s in self._boundary_stats])
        avg_min_prob = np.mean([s["min_prob"] for s in self._boundary_stats])
        avg_mean_logit = np.mean([s["mean_logit"] for s in self._boundary_stats])
        avg_max_logit = np.mean([s["max_logit"] for s in self._boundary_stats])
        avg_min_logit = np.mean([s["min_logit"] for s in self._boundary_stats])
        avg_pred_positive = np.mean([s["pred_positive_ratio"] for s in self._boundary_stats])
        avg_target_positive = np.mean([s["target_positive_ratio"] for s in self._boundary_stats])
        
        total_intersection = sum([s["intersection"] for s in self._boundary_stats])
        total_pred = sum([s["pred_total"] for s in self._boundary_stats])
        total_target = sum([s["target_total"] for s in self._boundary_stats])
        
        logging.info(f"\nEpoch {epoch} - Boundary Prediction Diagnostics:")
        logging.info(f"  Dice Score (threshold={avg_threshold:.3f}): {avg_dice:.6f}")
        logging.info(f"  Dice Score (prob-based, no threshold): {avg_dice_prob:.6f}")
        logging.info(f"  Probabilities: mean={avg_mean_prob:.6f}, max={avg_max_prob:.6f}, min={avg_min_prob:.6f}")
        logging.info(f"  Logits: mean={avg_mean_logit:.6f}, max={avg_max_logit:.6f}, min={avg_min_logit:.6f}")
        logging.info(f"  Positive Ratio: pred={avg_pred_positive:.6f}, target={avg_target_positive:.6f}")
        logging.info(f"  Overlap: intersection={total_intersection:.0f}, pred_pixels={total_pred:.0f}, target_pixels={total_target:.0f}")
        
        if avg_mean_prob < 0.01:
            logging.warning(f"    Very low mean probability ({avg_mean_prob:.6f}) - predictions may be too conservative")
        if avg_max_prob < 0.5:
            logging.warning(f"    Max probability ({avg_max_prob:.6f}) < 0.5 - no predictions above threshold!")
        if avg_pred_positive < 1e-6:
            logging.warning(f"    No positive predictions (ratio={avg_pred_positive:.6f}) - all predictions are zero!")
        if abs(avg_dice - 0.1313) < 1e-4:
            logging.warning(f"    Dice score is exactly 0.1313 - possible calculation issue or constant predictions")
        
        if epoch > 1 and hasattr(self, '_prev_boundary_stats'):
            prev_avg_dice = np.mean([s["dice"] for s in self._prev_boundary_stats])
            dice_change = abs(avg_dice - prev_avg_dice)
            if dice_change < 1e-6:
                logging.warning(f"    Dice score unchanged from previous epoch (change={dice_change:.2e})")
            
            prev_mean_prob = np.mean([s["mean_prob"] for s in self._prev_boundary_stats])
            prob_change = abs(avg_mean_prob - prev_mean_prob)
            if prob_change < 1e-6:
                logging.warning(f"    Mean probability unchanged (change={prob_change:.2e}) - predictions may not be learning")
        
        self._prev_boundary_stats = self._boundary_stats.copy()

    def _get_loss_criterion(self):
        loss_name = self.settings.loss_criterion
        
        loss_map = {
            "BCEDiceLoss": lambda: BCEDiceLoss(self.settings.alpha, self.settings.beta),
            "DiceLoss": lambda: DiceLoss(normalization="none"),
            "BCELoss": lambda: nn.BCEWithLogitsLoss(),
            "CrossEntropyLoss": lambda: nn.CrossEntropyLoss(),
            "GeneralizedDiceLoss": lambda: GeneralizedDiceLoss(),
            "TverskyLoss": lambda: TverskyLoss(self.label_no + 1),
            "BoundaryDoULoss": lambda: BoundaryDoULoss(),
            "BoundaryDoUDiceLoss": lambda: BoundaryDoUDiceLoss(alpha=0.5, beta=0.5),
            "BoundaryDoULossV2": lambda: BoundaryDoULossV2(),
            "BoundaryLoss": lambda: BoundaryLoss(),
            # New class-weighted Dice losses
            "ClassWeightedDiceLoss": lambda: ClassWeightedDiceLoss(
                num_classes=self.label_no,
                weight_mode=getattr(self.settings, "dice_weight_mode", "inverse_sqrt_freq"),
                exclude_background=getattr(self.settings, "exclude_background_from_dice", True),
            ),
            "CombinedCEDiceLoss": lambda: CombinedCEDiceLoss(
                num_classes=self.label_no,
                alpha=getattr(self.settings, "ce_weight", 0.5),
                beta=getattr(self.settings, "dice_weight", 0.5),
                dice_weight_mode=getattr(self.settings, "dice_weight_mode", "inverse_sqrt_freq"),
                exclude_background=getattr(self.settings, "exclude_background_from_dice", True),
            ),
        }
        
        if loss_name in loss_map:
            logging.info(f"Using {loss_name}")
            return loss_map[loss_name]()
        else:
            logging.error(f"Unknown loss criterion: {loss_name}, exiting")
            sys.exit(1)

    def _get_eval_metric(self):
        metric_name = self.settings.eval_metric
        
        if metric_name == "MeanIoU":
            logging.info("Using MeanIoU")
            return MeanIoU()
        elif metric_name == "DiceCoefficient":
            logging.info("Using DiceCoefficient")
            return DiceCoefficient()
        else:
            logging.error(f"Unknown evaluation metric: {metric_name}, exiting")
            sys.exit(1)

    def _ensure_tuple_output(self, output) -> tuple:
        """Ensure model output is a tuple for consistent handling."""
        if isinstance(output, (list, tuple)):
            return tuple(output)
        return (output,)

    # MULTI-CLASS DICE COMPUTATION
    
    def _compute_multiclass_dice(
        self,
        pred: torch.Tensor,      
        target: torch.Tensor,    
        num_classes: int,
        exclude_background: bool = False,
        smooth: float = 1e-7,
    ) -> Tuple[List[float], float]:
        """
        Compute per-class Dice and macro-averaged Dice.
        
        Args:
            pred: Predicted class indices (B, H, W)
            target: Ground truth class indices (B, H, W)
            num_classes: Number of segmentation classes
            exclude_background: Whether to exclude class 0 from averaging
            smooth: Smoothing factor
        
        Returns:
            dice_per_class: List of Dice scores per class (NaN for absent classes)
            mean_dice: Macro-averaged Dice score
        """
        dice_per_class = []
        
        start_class = 1 if exclude_background else 0
        
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            
            if union < smooth:
                # No pixels of this class in pred or target
                dice_c = float('nan')
            else:
                dice_c = (2.0 * intersection + smooth) / (union + smooth)
                dice_c = dice_c.item()
            
            dice_per_class.append(dice_c)
        
        # Macro average over classes that are present
        valid_dice = [
            d for i, d in enumerate(dice_per_class) 
            if i >= start_class and not math.isnan(d)
        ]
        
        mean_dice = sum(valid_dice) / len(valid_dice) if valid_dice else 0.0
        
        return dice_per_class, mean_dice
    
    def _compute_weighted_multiclass_dice(
        self,
        pred: torch.Tensor,      
        target: torch.Tensor,  
        num_classes: int,
        weight_mode: str = "inverse_sqrt_freq",
        exclude_background: bool = False,
        smooth: float = 1e-7,
    ) -> Tuple[List[float], float, List[float]]:
        """
        Compute per-class Dice with weighted averaging.
        
        Args:
            pred: Predicted class indices (B, H, W)
            target: Ground truth class indices (B, H, W)
            num_classes: Number of segmentation classes
            weight_mode: 'uniform', 'inverse_freq', 'inverse_sqrt_freq', 'pixel_count'
            exclude_background: Whether to exclude class 0
            smooth: Smoothing factor
        
        Returns:
            dice_per_class: List of Dice scores per class
            weighted_mean_dice: Weighted average Dice
            weights_used: Weights applied to each class
        """
        dice_per_class = []
        class_pixel_counts = []
        
        start_class = 1 if exclude_background else 0
        
        # Compute per-class Dice and pixel counts
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            
            intersection = (pred_c * target_c).sum()
            pred_sum = pred_c.sum()
            target_sum = target_c.sum()
            union = pred_sum + target_sum
            
            class_pixel_counts.append(target_sum.item())
            
            if union < smooth:
                dice_c = float('nan')
            else:
                dice_c = (2.0 * intersection + smooth) / (union + smooth)
                dice_c = dice_c.item()
            
            dice_per_class.append(dice_c)
        
        # Compute weights
        total_pixels = sum(class_pixel_counts[start_class:])
        weights = []
        
        for c in range(num_classes):
            if c < start_class:
                weights.append(0.0)
                continue
            
            freq = class_pixel_counts[c] / (total_pixels + smooth)
            
            if weight_mode == "inverse_freq":
                w = 1.0 / (freq + smooth) if freq > 0 else 0.0
            elif weight_mode == "inverse_sqrt_freq":
                w = 1.0 / (math.sqrt(freq) + smooth) if freq > 0 else 0.0
            elif weight_mode == "pixel_count":
                # Weight by pixel count (micro-like)
                w = class_pixel_counts[c]
            else:
                # Uniform
                w = 1.0 if class_pixel_counts[c] > 0 else 0.0
            
            weights.append(w)
        
        # Normalize 
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        
        # Compute weighted average (for valid Dice scores)
        weighted_sum = 0.0
        weight_used_sum = 0.0
        
        for c in range(num_classes):
            if not math.isnan(dice_per_class[c]) and weights[c] > 0:
                weighted_sum += dice_per_class[c] * weights[c]
                weight_used_sum += weights[c]
        
        weighted_mean_dice = weighted_sum / weight_used_sum if weight_used_sum > 0 else 0.0
        
        return dice_per_class, weighted_mean_dice, weights

    def _compute_eval_metrics(
        self, 
        outputs: tuple, 
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for all tasks with proper multi-class handling.
        
        Returns dict with 'seg_dice', per-class dice, and optionally 'boundary_dice'.
        """
        metrics = {}
        
        # Segmentation Metrics
        seg_output = outputs[0]
        if isinstance(targets, dict):
            seg_target = targets["seg"]
        else:
            seg_target = targets
        
        # Hard predictions for evaluation
        seg_probs = F.softmax(seg_output, dim=1)
        seg_pred = torch.argmax(seg_probs, dim=1) 
        seg_gt = torch.argmax(seg_target, dim=1)    
        
        # Compute multi-class Dice with weights
        if self.dice_averaging == "weighted":
            dice_per_class, mean_dice, weights = self._compute_weighted_multiclass_dice(
                seg_pred, seg_gt,
                num_classes=self.label_no,
                weight_mode=getattr(self.settings, "dice_weight_mode", "inverse_sqrt_freq"),
                exclude_background=self.exclude_background_from_dice,
            )
        else:
            # Macro averaging (default)
            dice_per_class, mean_dice = self._compute_multiclass_dice(
                seg_pred, seg_gt,
                num_classes=self.label_no,
                exclude_background=self.exclude_background_from_dice,
            )
        
        metrics["seg_dice"] = mean_dice
        
        # Store per-class Dice
        for c, d in enumerate(dice_per_class):
            metrics[f"dice_class_{c}"] = d if not math.isnan(d) else 0.0
        
        # Boundary Metrics 
        if isinstance(targets, dict) and "boundary" in targets and len(outputs) > 1:
            boundary_output = outputs[1]
            boundary_target = targets["boundary"]
            
            # Handle channel mismatch
            if boundary_output.shape[1] != boundary_target.shape[1]:
                boundary_output = boundary_output[:, :boundary_target.shape[1], :, :]
            
            boundary_probs = torch.sigmoid(boundary_output)
            
            # Adaptive threshold for sparse boundaries
            target_positive_ratio = boundary_target.sum().item() / boundary_target.numel()
            
            if target_positive_ratio > 0 and target_positive_ratio < 0.1:
                prob_flat = boundary_probs.view(-1).cpu()
                if len(prob_flat) > 0:
                    k = max(1, int((1 - target_positive_ratio) * len(prob_flat)))
                    k = min(k, len(prob_flat))
                    threshold_val, _ = torch.kthvalue(prob_flat, k)
                    threshold = threshold_val.item()
                    threshold = max(0.5, min(0.95, threshold))
                else:
                    threshold = 0.5
            else:
                threshold = 0.5
            
            boundary_pred = (boundary_probs > threshold).float()
            
            # Dice
            boundary_pred_flat = boundary_pred.view(boundary_pred.shape[0], boundary_pred.shape[1], -1)
            boundary_target_flat = boundary_target.view(boundary_target.shape[0], boundary_target.shape[1], -1)
            
            intersection = (boundary_pred_flat * boundary_target_flat).sum(dim=2)
            pred_sum = boundary_pred_flat.sum(dim=2)
            target_sum = boundary_target_flat.sum(dim=2)
            union = pred_sum + target_sum
            
            dice_per_sample = (2.0 * intersection + 1e-7) / (union + 1e-7)
            boundary_dice = dice_per_sample.mean()
            
            # Probability-based Dice
            prob_flat = boundary_probs.view(boundary_pred.shape[0], boundary_pred.shape[1], -1)
            target_flat = boundary_target_flat
            prob_dice_numerator = (prob_flat * target_flat).sum(dim=2) * 2.0
            prob_dice_denominator = prob_flat.sum(dim=2) + target_flat.sum(dim=2)
            prob_dice = (prob_dice_numerator + 1e-7) / (prob_dice_denominator + 1e-7)
            prob_dice_mean = prob_dice.mean()
            
            # logging
            mean_prob = boundary_probs.mean().item()
            max_prob = boundary_probs.max().item()
            min_prob = boundary_probs.min().item()
            pred_positive_ratio = boundary_pred.sum().item() / boundary_pred.numel()
            
            mean_logit = boundary_output.mean().item()
            max_logit = boundary_output.max().item()
            min_logit = boundary_output.min().item()
            
            intersection_total = (boundary_pred * boundary_target).sum().item()
            pred_total = boundary_pred.sum().item()
            target_total = boundary_target.sum().item()
            
            if boundary_dice.is_cuda:
                boundary_dice = boundary_dice.cpu().detach().numpy()
                prob_dice_mean = prob_dice_mean.cpu().detach().numpy()
            
            metrics["boundary_dice"] = float(boundary_dice)
            metrics["boundary_dice_prob"] = float(prob_dice_mean)
            
            if not hasattr(self, '_boundary_stats'):
                self._boundary_stats = []
            
            self._boundary_stats.append({
                "dice": float(boundary_dice),
                "dice_prob": float(prob_dice_mean),
                "threshold": threshold,
                "mean_prob": mean_prob,
                "max_prob": max_prob,
                "min_prob": min_prob,
                "mean_logit": mean_logit,
                "max_logit": max_logit,
                "min_logit": min_logit,
                "pred_positive_ratio": pred_positive_ratio,
                "target_positive_ratio": target_positive_ratio,
                "intersection": intersection_total,
                "pred_total": pred_total,
                "target_total": target_total,
            })
        
        return metrics

    def train_model(
        self,
        output_path: Path,
        num_epochs: int,
        patience: int,
        create: bool = True,
        frozen: bool = False,
    ) -> None:
        """
        Performs training of model for a number of epochs with automatic LR finding.

        Args:
            output_path: Path to save model file to.
            num_epochs: Number of epochs to train the model for.
            patience: Epochs to wait while validation loss not improving before stopping.
            create: Whether to create a new model and optimizer from scratch.
            frozen: Whether to freeze encoder convolutional layers.
        """
        if create:
            self._create_model_and_optimiser(self.starting_lr, frozen=frozen)
            
            if self.full_weights_path:
                logging.info("Loading pretrained weights for encoder and decoder (for LR finder).")
                model_tuple = create_model_from_file_full_weights(
                    self.full_weights_path, 
                    self.model_struc_dict, 
                    device_num=self.model_device_num
                )
                self.model, self.num_labels, self.label_codes = model_tuple
            
            lr_to_use = self._run_lr_finder()
            
            self._create_model_and_optimiser(lr_to_use, frozen=frozen)
            
            if self.full_weights_path:
                logging.info("Loading pretrained weights for encoder and decoder.")
                model_tuple = create_model_from_file_full_weights(
                    self.full_weights_path,
                    self.model_struc_dict, 
                    device_num=self.model_device_num
                )
                self.model, self.num_labels, self.label_codes = model_tuple
                
                if self.use_sam:
                    base_optimizer = torch.optim.AdamW
                    self.optimizer = SAM(
                        self.model.parameters(), 
                        base_optimizer, 
                        lr=lr_to_use, 
                        adaptive=self.adaptive_sam
                    )
                else:
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr_to_use)

            if self.encoder_weights_path:
                logging.info("Loading encoder weights.")
                self._load_encoder_weights(self.encoder_weights_path)

            early_stopping = self._create_early_stopping(output_path, patience)
        else:
            self.starting_lr /= self.lr_reduce_factor
            self.end_lr /= self.lr_reduce_factor
            self.log_lr_ratio = self._calculate_log_lr_ratio()
            self._load_in_model_and_optimizer(
                self.starting_lr, output_path, frozen=frozen, optimizer=False
            )
            lr_to_use = self._run_lr_finder()
            min_loss = self._load_in_model_and_optimizer(
                self.starting_lr, output_path, frozen=frozen, optimizer=False
            )
            early_stopping = self._create_early_stopping(
                output_path, patience, best_score=-min_loss
            )

        lr_scheduler = self._create_oc_lr_scheduler(num_epochs, lr_to_use)

        # === Main Training Loop ===
        for epoch in range(1, num_epochs + 1):
            tic = time.perf_counter()
            logging.info(f"Epoch {epoch}/{num_epochs}")
            
            # --- Training Phase ---
            self.model.train()
            self.train_loss_tracker.clear()
            
            for batch in tqdm(
                self.training_loader,
                desc="Training",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                self._train_one_batch(lr_scheduler, batch)

            # --- Validation Phase ---
            self.model.eval()
            self.valid_loss_tracker.clear()
            if self.use_multitask:
                self._boundary_stats = []
            
            with torch.no_grad():
                for batch in tqdm(
                    self.validation_loader,
                    desc="Validation",
                    bar_format=cfg.TQDM_BAR_FORMAT,
                ):
                    self._validate_one_batch(batch)

            # Epoch Statistics 
            toc = time.perf_counter()
            self._log_epoch_statistics(epoch, toc - tic)
            
            # Boundary Prediction Diagnostics 
            if self.use_multitask and hasattr(self, '_boundary_stats') and self._boundary_stats:
                self._log_boundary_prediction_statistics(epoch)
                self._boundary_stats = []
            
            # Diagnostic Logging 
            if epoch == 1 or epoch % 5 == 0:
                self._log_gradient_statistics(epoch)
            
            if epoch == 1 or epoch % 5 == 0:
                self._log_parameter_statistics(epoch)

            # Early Stopping Check 
            current_valid_loss = self.epoch_history["valid_total"][-1]
            early_stopping(current_valid_loss, self.model, self.optimizer, self.codes)

            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                break

        self._load_in_weights(output_path)

    def _train_one_batch(self, lr_scheduler, batch) -> torch.Tensor:
        """Train on a single batch, handling both single and multi-task modes."""
        inputs, targets = utils.prepare_training_batch(
            batch, self.model_device_num, self.label_no
        )
        
        is_multitask = isinstance(targets, dict)
        
        if is_multitask and self.use_multitask:
            loss = self._train_multitask_batch(inputs, targets)
        elif is_multitask and not self.use_multitask:
            seg_targets = targets.get("seg", targets)
            loss = self._train_singletask_batch(inputs, seg_targets)
        else:
            loss = self._train_singletask_batch(inputs, targets)
        
        lr_scheduler.step()
        return loss

    def _train_singletask_batch(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Standard single-task training step."""
        if self.use_sam:
            self.optimizer.zero_grad()
            output = self._ensure_tuple_output(self.model(inputs))[0]
            
            if self.settings.loss_criterion == "CrossEntropyLoss":
                loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
            else:
                loss = self.loss_criterion(output, targets.float())
            
            loss.backward()
            self.optimizer.first_step(zero_grad=True)
            
            output = self._ensure_tuple_output(self.model(inputs))[0]
            if self.settings.loss_criterion == "CrossEntropyLoss":
                loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
            else:
                loss = self.loss_criterion(output, targets.float())
            
            loss.backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.zero_grad()
            output = self._ensure_tuple_output(self.model(inputs))[0]
            
            if self.settings.loss_criterion == "CrossEntropyLoss":
                loss = self.loss_criterion(output, torch.argmax(targets, dim=1))
            else:
                loss = self.loss_criterion(output, targets.float())
            
            loss.backward()
            self.optimizer.step()
        
        self.train_loss_tracker.append_losses({"total": loss.item(), "seg": loss.item()})
        return loss

    def _train_multitask_batch(
        self, 
        inputs: torch.Tensor, 
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Multi-task training step with individual loss tracking."""
        if self.use_sam:
            self.optimizer.zero_grad()
            outputs = self._ensure_tuple_output(self.model(inputs))
            losses = self.loss_calculator.compute(outputs, targets)
            losses["total"].backward()
            self.optimizer.first_step(zero_grad=True)
            
            outputs = self._ensure_tuple_output(self.model(inputs))
            losses = self.loss_calculator.compute(outputs, targets)
            losses["total"].backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.zero_grad()
            outputs = self._ensure_tuple_output(self.model(inputs))
            losses = self.loss_calculator.compute(outputs, targets)
            losses["total"].backward()
            self.optimizer.step()
        
        self.train_loss_tracker.append_losses({
            k: v.item() for k, v in losses.items()
        })
        
        if not hasattr(self, '_grad_log_counter'):
            self._grad_log_counter = 0
        self._grad_log_counter += 1
        
        return losses["total"]

    def _validate_one_batch(self, batch) -> None:
        """Validate on a single batch with metric computation."""
        inputs, targets = utils.prepare_training_batch(
            batch, self.model_device_num, self.label_no
        )
        
        outputs = self._ensure_tuple_output(self.model(inputs))
        is_multitask = isinstance(targets, dict)
        
        if is_multitask and self.use_multitask:
            losses = self.loss_calculator.compute(outputs, targets)
            self.valid_loss_tracker.append_losses({
                k: v.item() for k, v in losses.items()
            })
        else:
            seg_output = outputs[0]
            if isinstance(targets, dict):
                seg_target = targets["seg"]
            else:
                seg_target = targets
            
            if self.settings.loss_criterion == "CrossEntropyLoss":
                loss = self.loss_criterion(seg_output, torch.argmax(seg_target, dim=1))
            else:
                loss = self.loss_criterion(seg_output, seg_target.float())
            
            self.valid_loss_tracker.append_losses({"total": loss.item(), "seg": loss.item()})
        
        metrics = self._compute_eval_metrics(outputs, targets)
        self.valid_loss_tracker.append_metrics(metrics)

    def _log_epoch_statistics(self, epoch: int, elapsed_time: float) -> None:
        """Log and store epoch-level statistics with per-class Dice."""
        train_avgs = self.train_loss_tracker.get_average_losses()
        valid_avgs = self.valid_loss_tracker.get_average_losses()
        valid_metrics = self.valid_loss_tracker.get_average_metrics()
        
        # Store in history
        self.epoch_history["train_total"].append(train_avgs.get("total", 0))
        self.epoch_history["train_seg"].append(train_avgs.get("seg", 0))
        self.epoch_history["train_boundary"].append(train_avgs.get("boundary", 0))
        self.epoch_history["train_task3"].append(train_avgs.get("task3", 0))
        
        self.epoch_history["valid_total"].append(valid_avgs.get("total", 0))
        self.epoch_history["valid_seg"].append(valid_avgs.get("seg", 0))
        self.epoch_history["valid_boundary"].append(valid_avgs.get("boundary", 0))
        self.epoch_history["valid_task3"].append(valid_avgs.get("task3", 0))
        
        self.epoch_history["seg_dice"].append(valid_metrics.get("seg_dice", 0))
        self.epoch_history["boundary_dice"].append(valid_metrics.get("boundary_dice", 0))
        
        # Per-class Dice
        for c in range(self.label_no):
            key = f"dice_class_{c}"
            self.epoch_history[key].append(valid_metrics.get(key, 0))
        
        log_parts = [f"Epoch {epoch}"]
        
        train_str = f"Train Loss: {train_avgs.get('total', 0):.4f}"
        if self.use_multitask:
            train_str += f" (Seg: {train_avgs.get('seg', 0):.4f}"
            if "boundary" in train_avgs:
                train_str += f", Bound: {train_avgs.get('boundary', 0):.4f}"
            if "task3" in train_avgs:
                train_str += f", Task3: {train_avgs.get('task3', 0):.4f}"
            train_str += ")"
        log_parts.append(train_str)
        
        valid_str = f"Val Loss: {valid_avgs.get('total', 0):.4f}"
        if self.use_multitask:
            valid_str += f" (Seg: {valid_avgs.get('seg', 0):.4f}"
            if "boundary" in valid_avgs:
                valid_str += f", Bound: {valid_avgs.get('boundary', 0):.4f}"
            if "task3" in valid_avgs:
                valid_str += f", Task3: {valid_avgs.get('task3', 0):.4f}"
            valid_str += ")"
        log_parts.append(valid_str)
        
        # Mean Dice
        metric_str = f"Seg Dice: {valid_metrics.get('seg_dice', 0):.4f}"
        if "boundary_dice" in valid_metrics:
            metric_str += f", Bound Dice: {valid_metrics.get('boundary_dice', 0):.4f}"
        log_parts.append(metric_str)
        
        log_parts.append(f"Time: {elapsed_time:.1f}s")
        
        logging.info(" | ".join(log_parts))
        
        # Log per-class Dice breakdown
        class_dice_parts = []
        for c in range(self.label_no):
            class_name = self.codes.get(c, f"C{c}") if self.codes else f"C{c}"
            dice_val = valid_metrics.get(f"dice_class_{c}", 0)
            class_dice_parts.append(f"{class_name}: {dice_val:.3f}")
        
        logging.info(f"  Per-class Dice: {' | '.join(class_dice_parts)}")

    def _load_in_model_and_optimizer(
        self, learning_rate, output_path, frozen=False, optimizer=False
    ):
        self._create_model_and_optimiser(learning_rate, frozen=frozen)
        logging.info("Loading weights from saved checkpoint.")
        loss_val = self._load_in_weights(output_path, optimizer=optimizer)
        return loss_val

    def _load_in_weights(self, output_path, optimizer=False, gpu=True):
        map_location = f"cuda:{self.model_device_num}" if gpu else "cpu"
        model_dict = torch.load(output_path, map_location=map_location, weights_only=False)
        logging.info("Loading model weights.")
        self.model.load_state_dict(model_dict["model_state_dict"])
        if optimizer:
            logging.info("Loading optimizer weights.")
            self.optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        return model_dict.get("loss_val", np.inf)

    def _run_lr_finder(self):
        logging.info("Finding optimal learning rate.")
        lr_scheduler = self._create_exponential_lr_scheduler()
        lr_find_loss, lr_find_lr = self._lr_finder(lr_scheduler)
        lr_to_use = self._find_lr_from_graph(lr_find_loss, lr_find_lr)
        logging.info(f"Selected learning rate: {lr_to_use:.6f}")
        return lr_to_use

    def _lr_finder(self, lr_scheduler, smoothing=0.05):
        lr_find_loss = []
        lr_find_lr = []
        iters = 0

        self.model.train()
        logging.info(f"Running LR finder for {self.lr_find_epochs} epochs.")
        
        for i in range(self.lr_find_epochs):
            for batch in tqdm(
                self.training_loader,
                desc=f"LR Finder Epoch {i + 1}",
                bar_format=cfg.TQDM_BAR_FORMAT,
            ):
                loss = self._train_one_batch(lr_scheduler, batch)
                lr_step = self.optimizer.state_dict()["param_groups"][0]["lr"]
                lr_find_lr.append(lr_step)
                
                if iters == 0:
                    lr_find_loss.append(loss)
                else:
                    loss = smoothing * loss + (1 - smoothing) * lr_find_loss[-1]
                    lr_find_loss.append(loss)
                
                if loss > 1 and iters > len(self.training_loader) // 1.333:
                    break
                iters += 1

        if self.settings.plot_lr_graph:
            fig = tpl.figure()
            fig.plot(
                np.log10(lr_find_lr),
                lr_find_loss,
                width=50,
                height=30,
                xlabel="Log10 Learning Rate",
            )
            fig.show()

        return lr_find_loss, lr_find_lr

    @staticmethod
    def _find_lr_from_graph(lr_find_loss, lr_find_lr) -> float:
        """Find LR at steepest loss descent."""
        default_min_lr = cfg.DEFAULT_MIN_LR
        
        for i in range(len(lr_find_loss)):
            if lr_find_loss[i].is_cuda:
                lr_find_loss[i] = lr_find_loss[i].cpu()
            lr_find_loss[i] = lr_find_loss[i].detach().numpy()
        
        losses = np.array(lr_find_loss)
        try:
            gradients = np.gradient(losses)
            min_gradient = gradients.min()
            if min_gradient < 0:
                min_loss_grad_idx = gradients.argmin()
            else:
                logging.info(f"Min gradient ({min_gradient}) positive, using default LR.")
                return default_min_lr
        except Exception as e:
            logging.info(f"Gradient computation failed: {e}. Using default LR.")
            return default_min_lr
        
        min_lr = lr_find_lr[min_loss_grad_idx]
        return min_lr / cfg.LR_DIVISOR

    def _lr_exp_stepper(self, x):
        return math.exp(
            x * self.log_lr_ratio / (self.lr_find_epochs * len(self.training_loader))
        )

    def _create_optimizer(self, learning_rate):
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _create_exponential_lr_scheduler(self):
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_exp_stepper)

    def _create_oc_lr_scheduler(self, num_epochs, lr_to_use):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr_to_use,
            steps_per_epoch=len(self.training_loader),
            epochs=num_epochs,
            pct_start=self.settings.pct_lr_inc,
        )

    def _create_early_stopping(self, output_path, patience, best_score=None):
        return EarlyStopping(
            patience=patience,
            verbose=True,
            path=output_path,
            model_dict=self.model_struc_dict,
            best_score=best_score,
        )

    def output_loss_fig(self, model_out_path: Path) -> None:
        """Save figures showing training/validation losses and metrics per task."""
        output_dir = model_out_path.parent
        epochs = range(1, len(self.epoch_history["train_total"]) + 1)
        
        if self.use_multitask:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Total Loss
            ax = axes[0, 0]
            ax.plot(epochs, self.epoch_history["train_total"], label="Train Total")
            ax.plot(epochs, self.epoch_history["valid_total"], label="Val Total")
            min_idx = np.argmin(self.epoch_history["valid_total"]) + 1
            ax.axvline(min_idx, linestyle="--", color="r", label="Best Checkpoint")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Total Loss")
            ax.legend()
            ax.grid(True)
            
            # Segmentation Loss
            ax = axes[0, 1]
            ax.plot(epochs, self.epoch_history["train_seg"], label="Train Seg")
            ax.plot(epochs, self.epoch_history["valid_seg"], label="Val Seg")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Segmentation Loss")
            ax.legend()
            ax.grid(True)
            
            # Boundary Loss
            ax = axes[0, 2]
            if any(self.epoch_history["train_boundary"]):
                ax.plot(epochs, self.epoch_history["train_boundary"], label="Train Boundary")
                ax.plot(epochs, self.epoch_history["valid_boundary"], label="Val Boundary")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Boundary Loss")
            ax.legend()
            ax.grid(True)
            
            # Mean Dice Metrics
            ax = axes[1, 0]
            ax.plot(epochs, self.epoch_history["seg_dice"], label="Seg Dice (mean)")
            if any(self.epoch_history["boundary_dice"]):
                ax.plot(epochs, self.epoch_history["boundary_dice"], label="Boundary Dice")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Dice Score")
            ax.set_title("Mean Evaluation Metrics")
            ax.legend()
            ax.grid(True)
            
            # Per-class Dice
            ax = axes[1, 1]
            for c in range(self.label_no):
                key = f"dice_class_{c}"
                if key in self.epoch_history and any(self.epoch_history[key]):
                    class_name = self.codes.get(c, f"Class {c}") if self.codes else f"Class {c}"
                    ax.plot(epochs, self.epoch_history[key], label=class_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Dice Score")
            ax.set_title("Per-Class Dice")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True)
            
            # Per-class Dice bar chart
            ax = axes[1, 2]
            final_dice = []
            class_names = []
            for c in range(self.label_no):
                key = f"dice_class_{c}"
                if key in self.epoch_history and self.epoch_history[key]:
                    final_dice.append(self.epoch_history[key][-1])
                    class_names.append(self.codes.get(c, f"C{c}") if self.codes else f"C{c}")
            
            if final_dice:
                bars = ax.bar(class_names, final_dice, color='steelblue')
                ax.axhline(np.mean(final_dice), color='r', linestyle='--', label=f"Mean: {np.mean(final_dice):.3f}")
                ax.set_xlabel("Class")
                ax.set_ylabel("Final Dice Score")
                ax.set_title("Final Per-Class Dice")
                ax.legend()
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, val in zip(bars, final_dice):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.suptitle(f"Training History: {model_out_path.stem}", fontsize=14)
            plt.tight_layout()
        else:
            # Single-task: 2x2 layout
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Loss
            ax = axes[0, 0]
            ax.plot(epochs, self.epoch_history["train_total"], label="Training Loss")
            ax.plot(epochs, self.epoch_history["valid_total"], label="Validation Loss")
            min_idx = np.argmin(self.epoch_history["valid_total"]) + 1
            ax.axvline(min_idx, linestyle="--", color="r", label="Best Checkpoint")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_title("Training & Validation Loss")
            ax.legend()
            ax.grid(True)
            
            # Mean Dice
            ax = axes[0, 1]
            ax.plot(epochs, self.epoch_history["seg_dice"], label="Mean Dice", color='green')
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Dice Score")
            ax.set_title("Mean Dice Score")
            ax.legend()
            ax.grid(True)
            
            # Per-class Dice curves
            ax = axes[1, 0]
            for c in range(self.label_no):
                key = f"dice_class_{c}"
                if key in self.epoch_history and any(self.epoch_history[key]):
                    class_name = self.codes.get(c, f"Class {c}") if self.codes else f"Class {c}"
                    ax.plot(epochs, self.epoch_history[key], label=class_name)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Dice Score")
            ax.set_title("Per-Class Dice Over Training")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True)
            
            # Per-class Dice bar chart
            ax = axes[1, 1]
            final_dice = []
            class_names = []
            for c in range(self.label_no):
                key = f"dice_class_{c}"
                if key in self.epoch_history and self.epoch_history[key]:
                    final_dice.append(self.epoch_history[key][-1])
                    class_names.append(self.codes.get(c, f"C{c}") if self.codes else f"C{c}")
            
            if final_dice:
                bars = ax.bar(class_names, final_dice, color='steelblue')
                ax.axhline(np.mean(final_dice), color='r', linestyle='--', 
                          label=f"Mean: {np.mean(final_dice):.3f}")
                ax.set_xlabel("Class")
                ax.set_ylabel("Final Dice Score")
                ax.set_title("Final Per-Class Dice Scores")
                ax.legend()
                ax.set_ylim(0, 1)
                
                for bar, val in zip(bars, final_dice):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.suptitle(f"Training History: {model_out_path.stem}", fontsize=14)
            plt.tight_layout()
        
        fig_out_path = output_dir / f"{model_out_path.stem}_loss_plot.png"
        logging.info(f"Saving loss figure to {fig_out_path}")
        fig.savefig(fig_out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        
        # Output CSV with all stats
        csv_path = output_dir / f"{model_out_path.stem}_train_stats.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Build header
            header = ["Epoch"]
            if self.use_multitask:
                header.extend([
                    "Train_Total", "Train_Seg", "Train_Boundary", "Train_Task3",
                    "Valid_Total", "Valid_Seg", "Valid_Boundary", "Valid_Task3",
                    "Seg_Dice", "Boundary_Dice"
                ])
            else:
                header.extend(["Train_Loss", "Valid_Loss", "Seg_Dice"])
            
            # Add per-class Dice columns
            for c in range(self.label_no):
                class_name = self.codes.get(c, f"Class_{c}") if self.codes else f"Class_{c}"
                header.append(f"Dice_{class_name}")
            
            writer.writerow(header)
            
            # Write data rows
            for i, epoch in enumerate(epochs):
                row = [epoch]
                
                if self.use_multitask:
                    row.extend([
                        self.epoch_history["train_total"][i],
                        self.epoch_history["train_seg"][i],
                        self.epoch_history["train_boundary"][i],
                        self.epoch_history["train_task3"][i],
                        self.epoch_history["valid_total"][i],
                        self.epoch_history["valid_seg"][i],
                        self.epoch_history["valid_boundary"][i],
                        self.epoch_history["valid_task3"][i],
                        self.epoch_history["seg_dice"][i],
                        self.epoch_history["boundary_dice"][i],
                    ])
                else:
                    row.extend([
                        self.epoch_history["train_total"][i],
                        self.epoch_history["valid_total"][i],
                        self.epoch_history["seg_dice"][i],
                    ])
                
                # Per-class Dice
                for c in range(self.label_no):
                    key = f"dice_class_{c}"
                    row.append(self.epoch_history[key][i] if key in self.epoch_history else 0)
                
                writer.writerow(row)
        
        logging.info(f"Saved training statistics to {csv_path}")

    def output_prediction_figure(self, model_path: Path) -> None:
        """Save visualization of predictions on validation samples."""
        self.model.eval()
        batch = next(iter(self.validation_loader))
        
        with torch.no_grad():
            inputs, targets = utils.prepare_training_batch(
                batch, self.model_device_num, self.label_no
            )
            outputs = self._ensure_tuple_output(self.model(inputs))
            
            seg_output = outputs[0]
            s_max = nn.Softmax(dim=1)
            probs = s_max(seg_output)
            seg_preds = torch.argmax(probs, dim=1)
            
            if isinstance(targets, dict):
                seg_target = targets.get("seg", None)
                boundary_target = targets.get("boundary", None)
                task3_target = targets.get("task3", None)
            else:
                seg_target = targets
                boundary_target = None
                task3_target = None
            
            seg_gt = None
            if seg_target is not None:
                seg_gt = torch.argmax(seg_target, dim=1)
            
            boundary_preds = None
            has_boundary_output = len(outputs) > 1
            if has_boundary_output:
                boundary_output = outputs[1]
                if boundary_target is not None:
                    if boundary_output.shape[1] != boundary_target.shape[1]:
                        boundary_output = boundary_output[:, :boundary_target.shape[1], :, :]
                else:
                    if boundary_output.shape[1] > 1:
                        boundary_output = boundary_output[:, 0:1, :, :]
                boundary_preds = (torch.sigmoid(boundary_output) > 0.5).float()
            
            task3_preds = None
            has_task3_output = len(outputs) > 2
            if has_task3_output:
                task3_output = outputs[2]
                if task3_target is not None:
                    if task3_output.shape[1] != task3_target.shape[1]:
                        task3_output = task3_output[:, :task3_target.shape[1], :, :]
                else:
                    if task3_output.shape[1] > 1:
                        task3_output = task3_output[:, 0:1, :, :]
                task3_preds = (torch.sigmoid(task3_output) > 0.5).float()

        bs = min(self.validation_loader.batch_size, 4)
        num_cols = 3
        
        if has_boundary_output:
            num_cols += 2
        if has_task3_output:
            num_cols += 2
        
        fig, axes = plt.subplots(bs, num_cols, figsize=(4 * num_cols, 4 * bs))
        if bs == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(bs):
            col_idx = 0
            img = inputs[i].cpu()
            
            if len(img.shape) == 4:
                img = img.squeeze(0)
            num_channels = img.shape[0]
            
            if num_channels == 3:
                img_display = img.permute(1, 2, 0)
                axes[i, col_idx].imshow(img_display)
            elif num_channels > 3:
                center = num_channels // 2
                axes[i, col_idx].imshow(img[center], cmap="gray")
            else:
                axes[i, col_idx].imshow(img.squeeze(), cmap="gray")
            if i == 0:
                axes[i, col_idx].set_title("Input")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            if seg_gt is not None:
                axes[i, col_idx].imshow(seg_gt[i].cpu(), cmap="tab10", vmin=0, vmax=self.label_no - 1)
                if i == 0:
                    axes[i, col_idx].set_title("Seg GT")
            else:
                axes[i, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
                if i == 0:
                    axes[i, col_idx].set_title("Seg GT")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            axes[i, col_idx].imshow(seg_preds[i].cpu(), cmap="tab10", vmin=0, vmax=self.label_no - 1)
            if i == 0:
                axes[i, col_idx].set_title("Seg Pred")
            axes[i, col_idx].axis("off")
            col_idx += 1
            
            if has_boundary_output:
                if boundary_target is not None:
                    b_gt = boundary_target[i, 0].cpu() if boundary_target.dim() == 4 else boundary_target[i].cpu()
                    axes[i, col_idx].imshow(b_gt, cmap="gray")
                    if i == 0:
                        axes[i, col_idx].set_title("Boundary GT")
                else:
                    axes[i, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
                    if i == 0:
                        axes[i, col_idx].set_title("Boundary GT")
                axes[i, col_idx].axis("off")
                col_idx += 1
                
                b_pred = boundary_preds[i, 0].cpu() if boundary_preds.dim() == 4 else boundary_preds[i].cpu()
                axes[i, col_idx].imshow(b_pred, cmap="gray")
                if i == 0:
                    axes[i, col_idx].set_title("Boundary Pred")
                axes[i, col_idx].axis("off")
                col_idx += 1
            
            if has_task3_output:
                if task3_target is not None:
                    t3_gt = task3_target[i, 0].cpu() if task3_target.dim() == 4 else task3_target[i].cpu()
                    axes[i, col_idx].imshow(t3_gt, cmap="gray")
                    if i == 0:
                        axes[i, col_idx].set_title("Task3 GT")
                else:
                    axes[i, col_idx].text(0.5, 0.5, "N/A", ha="center", va="center")
                    if i == 0:
                        axes[i, col_idx].set_title("Task3 GT")
                axes[i, col_idx].axis("off")
                col_idx += 1
                
                t3_pred = task3_preds[i, 0].cpu() if task3_preds.dim() == 4 else task3_preds[i].cpu()
                axes[i, col_idx].imshow(t3_pred, cmap="gray")
                if i == 0:
                    axes[i, col_idx].set_title("Task3 Pred")
                axes[i, col_idx].axis("off")
                col_idx += 1
        
        plt.suptitle(f"Predictions: {model_path.name}", fontsize=14)
        plt.tight_layout()
        
        out_path = model_path.parent / f"{model_path.stem}_prediction_image.png"
        logging.info(f"Saving prediction visualization to {out_path}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)