import logging
from pathlib import Path
from types import SimpleNamespace

from itertools import permutations as perm

import numpy as np
from scipy.stats import entropy
import torch
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from torch import nn as nn
from tqdm import tqdm
from volume_segmantics.data.dataloaders import get_2d_prediction_dataloader
from volume_segmantics.model.model_2d import create_model_from_file
from volume_segmantics.utilities.base_data_utils import Axis


class VolSeg2dPredictor:
    """Class that performs U-Net prediction operations. Does not interact with disk."""

    def __init__(self, model_file_path: str, settings: SimpleNamespace) -> None:
        self.model_file_path = Path(model_file_path)
        self.settings = settings
        self.model_device_num = int(settings.cuda_device)
        model_tuple = create_model_from_file(
            self.model_file_path, self.model_device_num
        )
        self.model, self.num_labels, self.label_codes = model_tuple

    def _get_model_from_trainer(self, trainer):
        self.model = trainer.model

    def _predict_single_axis(self, data_vol, output_probs=True, axis=Axis.Z):
        output_vol_list = []
        output_prob_list = []
        data_vol = utils.rotate_array_to_axis(data_vol, axis)
        yx_dims = list(data_vol.shape[1:])
        s_max = nn.Softmax(dim=1)
        data_loader = get_2d_prediction_dataloader(data_vol, self.settings)
        self.model.eval()
        logging.info(f"Predicting segmentation for volume of shape {data_vol.shape}.")
        with torch.no_grad():
            for batch in tqdm(
                data_loader, desc="Prediction batch", bar_format=cfg.TQDM_BAR_FORMAT
            ):
                output = self.model(batch.to(self.model_device_num))  # Forward pass
                probs = s_max(output)  # Convert the logits to probs
                # TODO: Don't flatten channels if one-hot output is needed
                labels = torch.argmax(probs, dim=1)  # flatten channels
                labels = utils.crop_tensor_to_array(labels, yx_dims)
                output_vol_list.append(labels.astype(np.uint8))
                if output_probs:
                    # Get indices of max probs
                    max_prob_idx = torch.argmax(probs, dim=1, keepdim=True)
                    # Extract along axis from outputs
                    probs = torch.gather(probs, 1, max_prob_idx)
                    # Remove the label dimension
                    probs = torch.squeeze(probs, dim=1)
                    probs = utils.crop_tensor_to_array(probs, yx_dims)
                    output_prob_list.append(probs.astype(np.float16))

        labels = np.concatenate(output_vol_list)
        labels = utils.rotate_array_to_axis(labels, axis)
        probs = np.concatenate(output_prob_list) if output_prob_list else None
        if probs is not None:
            probs = utils.rotate_array_to_axis(probs, axis)
        return labels, probs

    def _predict_3_ways_max_probs(self, data_vol):
        shape_tup = data_vol.shape
        logging.info("Creating empty data volumes in RAM to combine 3 axis prediction.")
        label_container = np.empty((2, *shape_tup), dtype=np.uint8)
        prob_container = np.empty((2, *shape_tup), dtype=np.float16)
        logging.info("Predicting YX slices:")
        label_container[0], prob_container[0] = self._predict_single_axis(
            data_vol, output_probs=True
        )
        logging.info("Predicting ZX slices:")
        label_container[1], prob_container[1] = self._predict_single_axis(
            data_vol, output_probs=True, axis=Axis.Y
        )
        logging.info("Merging XY and ZX volumes.")
        self._merge_vols_in_mem(prob_container, label_container)
        logging.info("Predicting ZY slices:")
        label_container[1], prob_container[1] = self._predict_single_axis(
            data_vol, output_probs=True, axis=Axis.X
        )
        logging.info("Merging max of XY and ZX volumes with ZY volume.")
        self._merge_vols_in_mem(prob_container, label_container)
        return label_container[0], prob_container[0]

    def _merge_vols_in_mem(self, prob_container, label_container):
        max_prob_idx = np.argmax(prob_container, axis=0)
        max_prob_idx = max_prob_idx[np.newaxis, :, :, :]
        prob_container[0] = np.squeeze(
            np.take_along_axis(prob_container, max_prob_idx, axis=0)
        )
        label_container[0] = np.squeeze(
            np.take_along_axis(label_container, max_prob_idx, axis=0)
        )

    def _predict_12_ways_max_probs(self, data_vol):
        shape_tup = data_vol.shape
        logging.info("Creating empty data volumes in RAM to combine 12 way prediction.")
        label_container = np.empty((2, *shape_tup), dtype=np.uint8)
        prob_container = np.empty((2, *shape_tup), dtype=np.float16)
        label_container[0], prob_container[0] = self._predict_3_ways_max_probs(data_vol)
        for k in range(1, 4):
            logging.info(f"Rotating volume {k * 90} degrees")
            data_vol = np.rot90(data_vol)
            labels, probs = self._predict_3_ways_max_probs(data_vol)
            label_container[1] = np.rot90(labels, -k)
            prob_container[1] = np.rot90(probs, -k)
            logging.info(
                f"Merging rot {k * 90} deg volume with rot {(k-1) * 90} deg volume."
            )
            self._merge_vols_in_mem(prob_container, label_container)
        return label_container[0], prob_container[0]

    def _predict_single_axis_to_one_hot(self, data_vol, axis=Axis.Z):
        prediction, _ = self._predict_single_axis(data_vol, axis=axis)
        return utils.one_hot_encode_array(prediction, self.num_labels)

    def _predict_3_ways_one_hot(self, data_vol):
        one_hot_out = self._predict_single_axis_to_one_hot(data_vol)
        one_hot_out += self._predict_single_axis_to_one_hot(data_vol, Axis.Y)
        one_hot_out += self._predict_single_axis_to_one_hot(data_vol, Axis.X)
        return one_hot_out

    def _predict_12_ways_one_hot(self, data_vol):
        one_hot_out = self._predict_3_ways_one_hot(data_vol)
        for k in range(1, 4):
            logging.info(f"Rotating volume {k * 90} degrees")
            data_vol = np.rot90(data_vol)
            one_hot_out += np.rot90(
                self._predict_3_ways_one_hot(data_vol), -k, axes=(-3, -2)
            )
        return one_hot_out

    def _predict_3_axis_generator(self, data_vol):
        for curr_axis in [Axis.Z, Axis.Y, Axis.X]:
            labels, _ = self._predict_single_axis(data_vol, output_probs=False, axis=curr_axis)
            yield labels

    def _predict_12_ways_generator(self, data_vol):
        for curr_axis in [Axis.Z, Axis.Y, Axis.X]:
            for k in range(4):
                labels, _ = self._predict_single_axis(np.rot90(data_vol, k), output_probs=False, axis=curr_axis)
                yield np.rot90(labels, -k)

    def _convert_labels_map_to_count(self, labels_vol):
        volume_size = labels_vol.shape
        num_labels = len(np.unique(labels_vol))

        counts_matrix = np.zeros((num_labels, *volume_size), dtype=np.uint8)
        for curr_label in range(num_labels):
            np.put(counts_matrix[curr_label],
                   np.argwhere(labels_vol.flatten()==curr_label),
                   1)

        return counts_matrix

    def _prediction_estimate_entropy(self, data_vol):
        if (self.settings.quality not in ["medium", "high"]):
            raise ValueError("Error in vol_seg_2d_predictor._prediction_estimate_entropy: Entropy calculation must be done with a minimum prediction quality of medium.")

        logging.info("Collecting voting distributions:")
        if self.settings.quality=="medium":
            g = self._predict_3_axis_generator(data_vol)
            labels = next(g)
            counts_matrix = self._convert_labels_map_to_count(labels)
            for _ in range(2):
                labels = next(g)
                counts_matrix += self._convert_labels_map_to_count(labels)
            probs_matrix = counts_matrix.astype(float) / 3

        else:
            g = self._predict_12_ways_generator(data_vol)

            labels = next(g)
            counts_matrix = self._convert_labels_map_to_count(labels)
            for _ in range(11):
                labels = next(g)
                counts_matrix += self._convert_labels_map_to_count(labels)
            probs_matrix = counts_matrix.astype(float) / 12

        logging.info("Aggregating prediction votes:")
        full_prediction_labels = np.argmax(counts_matrix, axis=0)
        full_prediction_probs = np.squeeze(
            np.take_along_axis(probs_matrix, full_prediction_labels[np.newaxis, ...], axis=0)
        )

        logging.info("Calculating prediction entropy (regularised) from voting distributions:")
        entropy_matrix = entropy(probs_matrix, axis=0)
        entropy_matrix /= entropy(np.full((len(np.unique(full_prediction_labels)),),
                                          1/len(np.unique(full_prediction_labels))))

        return full_prediction_labels, full_prediction_probs, entropy_matrix
