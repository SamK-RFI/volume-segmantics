import logging
from pathlib import Path
from typing import Tuple

import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import volume_segmantics.utilities.base_data_utils as utils
from torch.nn import DataParallel
import volume_segmantics.utilities.config as cfg
from volume_segmantics.model.vanilla_unet import UNet as VanillaUNet


def create_model_on_device(device_num: int, model_struc_dict: dict) -> torch.nn.Module:
    struct_dict_copy = model_struc_dict.copy()
    model_type = struct_dict_copy.pop("type")
    in_channels_requested = struct_dict_copy.get("in_channels", cfg.MODEL_INPUT_CHANNELS)
    encoder_weights = struct_dict_copy.get("encoder_weights", None)
    
    if model_type == utils.ModelType.U_NET:
        if struct_dict_copy['encoder_name'] == 'efficientnet-b5':
            import segmentation_models_pytorch as smp_old
            model = smp_old.Unet(**struct_dict_copy)
        elif struct_dict_copy['encoder_name'] in {'convnext_base', 'convnext_large', 'swin_base_patch4_window12_384'}:
            model = smp.Unet(**struct_dict_copy, encoder_depth=4,
                decoder_channels=(256, 128, 64, 32),
                head_upsampling=2,)
        else:
            model = smp.Unet(**struct_dict_copy)
        logging.info(f"Sending the U-Net model to device {device_num}")
    elif model_type == utils.ModelType.U_NET_PLUS_PLUS:
        if struct_dict_copy['encoder_name'] in {'convnext_base', 'convnext_large', 'swin_base_patch4_window12_384'}:
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
        logging.info(f"Sending the Linknet model to device {device_num}")
    elif model_type == utils.ModelType.PAN:
        model = smp.PAN(**struct_dict_copy)
        logging.info(f"Sending the PAN model to device {device_num}")
    elif model_type == utils.ModelType.SEGFORMER:
        model = smp.Segformer(**struct_dict_copy)
        logging.info(f"Sending the Segformer model to device {device_num}")
    elif model_type == utils.ModelType.VANILLA_UNET:
        in_channels = struct_dict_copy.get("in_channels", 1)
        model = VanillaUNet(in_channels=in_channels, out_classes=struct_dict_copy["classes"], up_sample_mode='conv_transpose')
        logging.info(f"Sending the Vanilla Unet model to device {device_num}")

    # If using pretrained weights with >3 input channels, adapt first conv by averaging RGB.
    try:
        if encoder_weights and encoder_weights != "None" and in_channels_requested > 3:
            _adapt_first_conv_from_imagenet_avg(model, model_type, struct_dict_copy, in_channels_requested)
    except Exception as e:
        logging.warning(f"Could not adapt first conv weights for in_channels={in_channels_requested}: {e}")

    if torch.cuda.device_count() > 1 and cfg.USE_ALL_GPUS:
        logging.info(f"Using {torch.cuda.device_count()} GPUs.")
        model = DataParallel(model)
        model.to("cuda")
    else:
        model.to(device_num)
    return model


def _adapt_first_conv_from_imagenet_avg(model: torch.nn.Module, model_type, struct_dict_copy: dict, in_channels_requested: int) -> None:
    """When using pretrained encoders and in_channels > 3, adapt the first conv layer weights by
    averaging the 3-channel Imagenet weights and repeating to match the requested channel count.
    This keeps the rest of the encoder on pretrained weights while providing a better init.
    """
    
    if not hasattr(model, 'encoder'):
        return
    encoder = model.encoder
    first_conv = None
    for m in encoder.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    if first_conv is None:
        return
    
    # Create a temporary 3-channel model with same architecture to pull pretrained RGB weights
    struct_rgb = struct_dict_copy.copy()
    struct_rgb['in_channels'] = 3
    model_type_local = model_type
    if model_type_local == utils.ModelType.U_NET:
        tmp = smp.Unet(**struct_rgb)
    elif model_type_local == utils.ModelType.U_NET_PLUS_PLUS:
        tmp = smp.UnetPlusPlus(**struct_rgb)
    elif model_type_local == utils.ModelType.FPN:
        tmp = smp.FPN(**struct_rgb)
    elif model_type_local == utils.ModelType.DEEPLABV3:
        tmp = smp.DeepLabV3(**struct_rgb)
    elif model_type_local == utils.ModelType.DEEPLABV3_PLUS:
        tmp = smp.DeepLabV3Plus(**struct_rgb)
    elif model_type_local == utils.ModelType.MA_NET:
        tmp = smp.MAnet(**struct_rgb)
    elif model_type_local == utils.ModelType.LINKNET:
        tmp = smp.Linknet(**struct_rgb)
    elif model_type_local == utils.ModelType.PAN:
        tmp = smp.PAN(**struct_rgb)
    elif model_type_local == utils.ModelType.SEGFORMER:
        tmp = smp.Segformer(**struct_rgb)
    else:
        return
    tmp_encoder = tmp.encoder
    tmp_first_conv = None
    for m in tmp_encoder.modules():
        if isinstance(m, nn.Conv2d):
            tmp_first_conv = m
            break
    if tmp_first_conv is None:
        return

    
    rgb_w = tmp_first_conv.weight.data  
    if rgb_w.shape[1] != 3:
        return
    mean_w = rgb_w.mean(dim=1, keepdim=True)
    new_w = mean_w.repeat(1, in_channels_requested, 1, 1)
    
    with torch.no_grad():
        if first_conv.weight.data.shape == new_w.shape:
            first_conv.weight.data.copy_(new_w.to(first_conv.weight.device).type(first_conv.weight.dtype))
        else:
            # If shapes differ do nothing
            return


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
    model_dict = torch.load(weights_fn, map_location=map_location, weights_only=False)
    model = create_model_on_device(device_num, model_dict["model_struc_dict"])
    logging.info("Loading in the saved weights.")
    model.load_state_dict(model_dict["model_state_dict"])
    return model, model_dict["model_struc_dict"]["classes"], model_dict["label_codes"]


def create_model_from_file_full_weights(
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
    num_classes = 2 #not used
    return model, num_classes, None

