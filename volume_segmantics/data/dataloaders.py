from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from torch.utils.data import DataLoader, Subset
from volume_segmantics.data.datasets import (get_2d_prediction_dataset,
                                             get_2d_training_dataset,
                                             get_2d_validation_dataset,
                                             get_2d_image_dir_prediction_dataset)

try:
    from monai.data import list_data_collate
    from volume_segmantics.data.datasets_monai import (
        get_monai_training_and_validation_datasets,
    )
    MONAI_DATASETS_AVAILABLE = True
except ImportError:
    MONAI_DATASETS_AVAILABLE = False
    list_data_collate = None


def get_2d_training_dataloaders(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:
    """Returns 2d training and validation dataloaders with indices split at random
    according to the percentage split specified in settings.

    Args:
        image_dir (Path): Directory of data images
        label_dir (Path): Directory of label images
        settings (SimpleNamespace): Settings object

    Returns:
        Tuple[DataLoader, DataLoader]: 2d training and validation dataloaders
    """
    use_monai = (
        getattr(settings, "augmentation_library", "albumentations") == "monai"
        and getattr(settings, "use_monai_datasets", True)
        and MONAI_DATASETS_AVAILABLE
    )

    if use_monai:
        return get_monai_training_dataloaders(image_dir, label_dir, settings)

    training_set_prop = settings.training_set_proportion
    batch_size = utils.get_batch_size(settings)

    full_training_dset = get_2d_training_dataset(image_dir, label_dir, settings)
    full_validation_dset = get_2d_validation_dataset(image_dir, label_dir, settings)
    # split the dataset into train and test 
    dset_length = len(full_training_dset)
    indices = torch.randperm(dset_length).tolist()
    train_idx, validate_idx = np.split(indices, [int(dset_length * training_set_prop)])
    training_dataset = Subset(full_training_dset, train_idx)
    validation_dataset = Subset(full_validation_dset, validate_idx)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )
    return training_dataloader, validation_dataloader


def get_monai_training_dataloaders(
    image_dir: Path, label_dir: Path, settings: SimpleNamespace
) -> Tuple[DataLoader, DataLoader]:
    """Returns MONAI-based training and validation dataloaders.

    Args:
        image_dir (Path): Directory of data images
        label_dir (Path): Directory of label images
        settings (SimpleNamespace): Settings object

    Returns:
        Tuple[DataLoader, DataLoader]: MONAI training and validation dataloaders
    """
    if not MONAI_DATASETS_AVAILABLE:
        raise ImportError(
            "MONAI datasets are not available. Install MONAI to use MONAI datasets."
        )

    batch_size = utils.get_batch_size(settings)

    training_dataset, validation_dataset = get_monai_training_and_validation_datasets(
        image_dir, label_dir, settings
    )

    # Create dataloaders with MONAI collate function
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=cfg.PIN_CUDA_MEMORY,
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )
    return training_dataloader, validation_dataloader


def get_2d_prediction_dataloader(
    data_vol: np.array, settings: SimpleNamespace
) -> DataLoader:
    pred_dataset = get_2d_prediction_dataset(data_vol, settings)
    batch_size = utils.get_batch_size(settings, prediction=True)
    return DataLoader(
        pred_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0, 
        pin_memory=cfg.PIN_CUDA_MEMORY,
    )


def get_2d_image_dir_prediction_dataloader(
    image_dir: Path, settings: SimpleNamespace
) -> DataLoader:
    pred_dataset = get_2d_image_dir_prediction_dataset(image_dir, settings)
    images_fps = pred_dataset.images_fps
    #batch_size = utils.get_batch_size(settings, prediction=True)
    return DataLoader(
        pred_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  
        pin_memory=cfg.PIN_CUDA_MEMORY,
    ), images_fps
