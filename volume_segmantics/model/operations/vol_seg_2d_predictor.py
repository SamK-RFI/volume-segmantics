import logging
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

from itertools import permutations as perm

import numpy as np
from scipy.stats import entropy
import sparse

import torch
import volume_segmantics.utilities.base_data_utils as utils
import volume_segmantics.utilities.config as cfg
from torch import nn as nn
from tqdm import tqdm
from volume_segmantics.data.dataloaders import get_2d_prediction_dataloader, get_2d_image_dir_prediction_dataloader
from volume_segmantics.model.model_2d import create_model_from_file
from volume_segmantics.utilities.base_data_utils import Axis
from volume_segmantics.data import get_settings_data

class VolSeg2dPredictor:
    """Class that performs U-Net prediction operations. Does not interact with disk."""

    def __init__(self, model_file_path: str, settings: SimpleNamespace) -> None:
        self.model_file_path = Path(model_file_path)
        self.settings = settings
        self.model_device_num = int(settings.cuda_device)
        model_tuple = create_model_from_file(
            self.model_file_path, device_num=self.model_device_num
        )
        self.model, self.num_labels, self.label_codes = model_tuple

    def _get_model_from_trainer(self, trainer):
        self.model = trainer.model

    def _predict_single_axis(self, data_vol, output_probs=True, axis=Axis.Z):
        output_vol_list = []
        output_prob_list = []
        output_logits_list = []
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

                    #logits = torch.squeeze(output, dim=1)
                    #logits = output[:,0,:]
                    logits = utils.crop_tensor_to_array(output, yx_dims)
                    output_logits_list.append(logits)

        labels = np.concatenate(output_vol_list)
        labels = utils.rotate_array_to_axis(labels, axis)
        probs = np.concatenate(output_prob_list) if output_prob_list else None
        logits = np.concatenate(output_logits_list) if output_logits_list else None

        if probs is not None:
            probs = utils.rotate_array_to_axis(probs, axis)

        if logits is not None:
            logits = utils.rotate_array_to_axis(logits, axis)

        return labels, probs, logits

    def _predict_3_ways_max_probs(self, data_vol):
        shape_tup = data_vol.shape
        logging.info("Creating empty data volumes in RAM to combine 3 axis prediction.")
        label_container = np.empty((2, *shape_tup), dtype=np.uint8)
        prob_container = np.empty((2, *shape_tup), dtype=np.float16)
        logging.info("Predicting YX slices:")
        label_container[0], prob_container[0],_ = self._predict_single_axis(
            data_vol, output_probs=True
        )
        logging.info("Predicting ZX slices:")
        label_container[1], prob_container[1],_ = self._predict_single_axis(
            data_vol, output_probs=True, axis=Axis.Y
        )
        logging.info("Merging XY and ZX volumes.")
        self._merge_vols_in_mem(prob_container, label_container)
        logging.info("Predicting ZY slices:")
        label_container[1], prob_container[1],_ = self._predict_single_axis(
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

    def _predict_Zonly_max_probs(self, data_vol):
        shape_tup = data_vol.shape
        logging.info("Creating empty data volumes in RAM to combine 12 way prediction.")
        label_container = np.empty((2, *shape_tup), dtype=np.uint8)
        prob_container = np.empty((2, *shape_tup), dtype=np.float16)
        label_container[0], prob_container[0] = self._predict_single_axis(data_vol)
        for k in range(1, 4):
            logging.info(f"Rotating volume {k * 90} degrees")
            data_vol = np.rot90(data_vol)
            labels, probs = self._predict_single_axis(data_vol)
            label_container[1] = np.rot90(labels, -k)
            prob_container[1] = np.rot90(probs, -k)
            logging.info(
                f"Merging rot {k * 90} deg volume with rot {(k-1) * 90} deg volume."
            )
            self._merge_vols_in_mem(prob_container, label_container)
        return label_container[0], prob_container[0]

    def _predict_single_axis_to_one_hot(self, data_vol, axis=Axis.Z):
        prediction, _, _ = self._predict_single_axis(data_vol, axis=axis)
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
            rotation_axes = {
                Axis.Z: (1, 2),
                Axis.X: (0, 2),
                Axis.Y: (0, 1)
            }
            for k in range(4):
                labels, probs = self._predict_single_axis(np.ascontiguousarray(np.rot90(data_vol, k, axes=rotation_axes[curr_axis])), output_probs=False, axis=curr_axis)
                yield np.rot90(labels, -k, axes=rotation_axes[curr_axis])

    def _predict_Zonly_generator(self, data_vol):
        for k in range(4):
            labels, probs = self._predict_single_axis(np.ascontiguousarray(np.rot90(data_vol, k, axes=(1, 2))), output_probs=False, axis=Axis.Z)
            yield np.rot90(labels, -k, axes=(1, 2))

    def _convert_labels_map_to_count(self, labels_vol):
        volume_size = labels_vol.shape
        logging.info(f"Label volume shape = {volume_size}")

        label_vol_contig = np.ascontiguousarray(labels_vol)
        label_counts = np.bincount(label_vol_contig.ravel())
        label_unique = np.nonzero(label_counts)[0]
        label_sorted = label_unique[np.argsort([i for i in label_counts if i>0])]

        logging.info(f"Unique labels: {label_sorted}")
        logging.info(f"Label counts: {np.sort([i for i in label_counts if i>0])}")

        label_flattened = labels_vol.flatten()
        counts_matrix = np.zeros((len(label_sorted), *volume_size), dtype=np.uint8)
        for idx, curr_label in tqdm(enumerate(label_sorted[:-1]), total=len(label_sorted[:-1])):
            np.put(counts_matrix[idx],
                   np.argwhere(label_flattened==curr_label),
                   1)
        counts_matrix[-1] = 1 - np.any(counts_matrix, axis=0)

        return counts_matrix, label_sorted

    def _prediction_estimate_entropy(self, data_vol):
        if (self.settings.quality not in ["medium", "high", "z_only"]):
            raise ValueError("Error in vol_seg_2d_predictor._prediction_estimate_entropy: Entropy calculation must be done with a minimum prediction quality of medium.")

        logging.info("Collecting voting distributions:")
        probs_matrix = np.zeros((self.num_labels, *data_vol.shape), dtype=np.uint8)
        if self.settings.quality=="medium":
            g = self._predict_3_ways_generator(data_vol)
            curr_counts, labels_list = self._convert_labels_map_to_count(data_vol)
            for i in range(3):
                logging.info(f"Voter {i+1} of 3 voting...")
                labels = next(g)
                logging.info(f"Converting votes...")
                curr_counts, labels_list = self._convert_labels_map_to_count(labels)
                for idx, curr_label in enumerate(curr_counts):
                    probs_matrix[labels_list[idx]] += curr_label

        elif self.settings.quality=="medium":
            g = self._predict_12_ways_generator(data_vol)
            for i in range(12):
                logging.info(f"Voter {i+1} of 12 voting...")
                labels = next(g)
                logging.info(f"Converting votes...")
                curr_counts, labels_list = self._convert_labels_map_to_count(labels)
                for idx, curr_label in enumerate(curr_counts):
                    probs_matrix[labels_list[idx]] += curr_label

        elif self.settings.quality=="z_only":
            g = self._predict_Zonly_generator(data_vol)
            for i in range(4):
                logging.info(f"Voter {i+1} of 4 voting...")
                labels = next(g)
                logging.info(f"Converting votes...")
                curr_counts, labels_list = self._convert_labels_map_to_count(labels)
                for idx, curr_label in enumerate(curr_counts):
                    probs_matrix[labels_list[idx]] += curr_label

        logging.info("Aggregating prediction votes:")
        probs_matrix_contig = np.ascontiguousarray(probs_matrix)
        full_prediction_labels = np.argmax(probs_matrix_contig, axis=0)
        full_prediction_probs = np.squeeze(
            np.take_along_axis(probs_matrix_contig, full_prediction_labels[np.newaxis, ...], axis=0)
        )

        if self.settings.quality=="medium":
            full_prediction_probs = full_prediction_probs.astype(float) / 3
        else:
            full_prediction_probs = full_prediction_probs.astype(float) / 12

        logging.info("Calculating prediction entropy (regularised) from voting distributions:")
        entropy_matrix = np.empty(data_vol.shape)
        for curr_slice in range(len(data_vol)):
            entropy_matrix[curr_slice] = entropy(probs_matrix_contig[:, curr_slice, ...], axis=0)
        entropy_matrix /= entropy(np.full((len(np.unique(full_prediction_labels)),),
                                          1/len(np.unique(full_prediction_labels))))

        return full_prediction_labels, full_prediction_probs, entropy_matrix




class VolSeg2dImageDirPredictor:
    """Class that performs U-Net prediction operations. Does not interact with disk."""

    def __init__(self, model_file_path: str, settings: SimpleNamespace) -> None:
        self.model_file_path = Path(model_file_path)
        self.settings = settings
        self.model_device_num = int(settings.cuda_device)
        model_tuple = create_model_from_file(
            self.model_file_path, device_num=self.model_device_num
        )
        self.model, self.num_labels, self.label_codes = model_tuple

    def _get_model_from_trainer(self, trainer):
        self.model = trainer.model

    def _predict_image_dir(self, image_dir, output_probs=False):
        output_vol_list = []
        output_prob_list = []
        #data_vol = utils.rotate_array_to_axis(data_vol, axis)
        #yx_dims = list(data_vol.shape[1:])
        print(self.settings)
        yx_dims = (self.settings.output_size, self.settings.output_size)

        s_max = nn.Softmax(dim=1)
        data_loader, images_fps = get_2d_image_dir_prediction_dataloader(image_dir, self.settings)
        self.model.eval()
        logging.info(f"Predicting segmentation for image dir.")
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


        return output_vol_list, output_prob_list, images_fps

    # TODO FIX
    def _predict_image_dir_to_one_hot(self, data_vol):
        prediction, _, images_fps = self._predict_single_axis(data_vol)
        return utils.one_hot_encode_array(prediction, self.num_labels), images_fps

