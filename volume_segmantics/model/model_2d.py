import logging
from pathlib import Path
from typing import Tuple

import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import volume_segmantics.utilities.base_data_utils as utils
from torch.nn import DataParallel

def create_model_on_device(device_num: int, model_struc_dict: dict) -> torch.nn.Module:
    struct_dict_copy = model_struc_dict.copy()
    model_type = struct_dict_copy.pop("type")
    
    if model_type == utils.ModelType.U_NET:

        if struct_dict_copy['encoder_name'] == 'efficientnet-b5':
            import segmentation_models_pytorch as smp_old

            model = smp_old.Unet(**struct_dict_copy)
        if struct_dict_copy['encoder_name'] in {'convnext_base', 'convnext_large', 'swin_base_patch4_window12_384'}:    #model = smp.Unet(**struct_dict_copy)
            model = smp.Unet(**struct_dict_copy, encoder_depth=4,
                decoder_channels=(256, 128, 64, 32),
                head_upsampling=2,)
        else:
            model = smp.Unet(**struct_dict_copy)
        logging.info(f"Sending the U-Net model to device {device_num}")
    elif model_type == utils.ModelType.U_NET_PLUS_PLUS:
        
        if struct_dict_copy['encoder_name'] in {'convnext_base', 'convnext_large', 'swin_base_patch4_window12_384'}:    #model = smp.Unet(**struct_dict_copy)
            model = smp.UnetPlusPlus(**struct_dict_copy, encoder_depth=4,
                decoder_channels=(256, 128, 64, 32),
                head_upsampling=2,)
        else:
            model = smp.UnetPlusPlus(**struct_dict_copy)
        logging.info(f"Sending the U-Net++ model to device {device_num}")
    elif model_type == utils.ModelType.FPN:
        model = smp.FPN(**struct_dict_copy)
        logging.info(f"Sending the FPN model to device {device_num}")
    elif model_type == utils.ModelType.DEEPLABV3:
        model = smp.DeepLabV3(**struct_dict_copy)
        logging.info(f"Sending the DeepLabV3 model to device {device_num}")
    elif model_type == utils.ModelType.DEEPLABV3_PLUS:
        model = smp.DeepLabV3Plus(**struct_dict_copy)
        logging.info(f"Sending the DeepLabV3+ model to device {device_num}")
    elif model_type == utils.ModelType.MA_NET:
        model = smp.MAnet(**struct_dict_copy)
        logging.info(f"Sending the MA-Net model to device {device_num}")
    elif model_type == utils.ModelType.LINKNET:
        model = smp.Linknet(**struct_dict_copy)
    elif model_type == utils.ModelType.PAN:
        model = smp.PAN(**struct_dict_copy)
        logging.info(f"Sending the Linknet model to device {device_num}")
    elif model_type == utils.ModelType.SEGFORMER:
        model = smp.Segformer(**struct_dict_copy)
        logging.info(f"Sending the Segformer model to device {device_num}")

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs.")
        model = DataParallel(model)
        model.to("cuda")
    else:
        model.to(device_num)
    return model


def create_model_from_file(
    weights_fn: Path, gpu: bool = True, device_num: int = 0,
) -> Tuple[torch.nn.Module, int, dict]:
    """Creates and returns a model and the number of segmentation labels
    that are predicted by the model."""
    if gpu:
        map_location = f"cuda:{device_num}"
    else:
        map_location = "cpu"
    weights_fn = weights_fn.resolve()
    logging.info("Loading model dictionary from file.")
    model_dict = torch.load(weights_fn, map_location=map_location, weights_only, weights_only=False)
    model = create_model_on_device(device_num, model_dict["model_struc_dict"])
    logging.info("Loading in the saved weights.")
    model.load_state_dict(model_dict["model_state_dict"])
    return model, model_dict["model_struc_dict"]["classes"], model_dict["label_codes"]


def create_model_from_file2(
    weights_fn: Path, model_struc_dict : dict, device_num: int = 0, gpu: bool = True, 
) -> Tuple[torch.nn.Module, int, dict]:
    """Creates and returns a model and the number of segmentation labels
    that are predicted by the model."""
    if gpu:
        map_location = f"cuda:{device_num}"
    else:
        map_location = "cpu"
    weights_fn = weights_fn.resolve()
    logging.info("Loading model dictionary from file.")

    model = create_model_on_device(device_num, model_struc_dict)
    
    model_dict = torch.load(weights_fn, map_location=map_location, weights_only=False)
    
    logging.info("Loading in the saved weights.")
    model.load_state_dict(model_dict, strict=False)
            
    model.to(device=map_location)
    num_classes = 1
    
    return model, num_classes, None

