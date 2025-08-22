import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import volume_segmantics.utilities.base_data_utils as utils
from skimage import img_as_ubyte, io
from tqdm import tqdm
from volume_segmantics.data.base_data_manager import BaseDataManager
from typing import Union


class TrainingDataSlicer(BaseDataManager):
    """
    Class that performs image preprocessing and provides methods to 
    convert 3d data volumes into 2d image slices on disk for model training.
    Slicing is carried in all of the xy (z), xz (y) and yz (x) planes.
    Supports both 2D (single channel) and 2.5D (RGB) slicing modes.
    """

    def __init__(
        self,
        data_vol: Union[str, np.ndarray],
        label_vol: Union[str, np.ndarray],
        settings: SimpleNamespace,
    ):
        """Inits TrainingDataSlicer.

        Args:
            data_vol(Union[str, np.ndarray]): Either a path to an image data volume or a numpy array of 3D image data
            label_vol(Union[str, np.ndarray]): Either a path to a label data volume or a numpy array of 3D label data
            settings(SimpleNamespace): An object containing the training settings
        """
        super().__init__(data_vol, settings)
        self.data_im_out_dir = None
        self.seg_im_out_dir = None
        self.multilabel = False
        self.settings = settings
        
        self.use_2_5d_slicing = getattr(settings, 'use_2_5d_slicing', False)
        self.skip_border_slices = getattr(settings, 'skip_border_slices', False)
        
        if self.use_2_5d_slicing:
            logging.info("2.5D slicing mode enabled - creating RGB images from adjacent slices")
            if self.skip_border_slices:
                logging.info("Border slices will be skipped in 2.5D mode")
        
        self.label_vol_path = utils.setup_path_if_exists(label_vol)
        if self.label_vol_path is not None:
            self.seg_vol, _ = utils.get_numpy_from_path(
                self.label_vol_path, internal_path=settings.seg_hdf5_path
            )
        elif isinstance(label_vol, np.ndarray):
            self.seg_vol = label_vol
        self._preprocess_labels()

    def _preprocess_labels(self):
        seg_classes = np.unique(self.seg_vol)
        self.num_seg_classes = len(seg_classes)
        if self.num_seg_classes > 2:
            self.multilabel = True
        logging.info(
            "Number of classes in segmentation dataset:" f" {self.num_seg_classes}"
        )
        logging.info(f"These classes are: {seg_classes}")
        if seg_classes[0] != 0 or not utils.sequential_labels(seg_classes):
            logging.info("Fixing label classes.")
            self._fix_label_classes(seg_classes)
        self.codes = [f"label_val_{i}" for i in seg_classes]

    def _fix_label_classes(self, seg_classes):
        """Changes the data values of classes in a segmented volume so that
        they start from zero.

        Args:
            seg_classes(list): An ascending list of the labels in the volume.
        """
        for idx, current in enumerate(seg_classes):
            self.seg_vol[self.seg_vol == current] = idx

    def output_data_slices(self, data_dir: Path, prefix: str) -> None:
        """
        Method that triggers slicing image data volume to disk in the
        xy (z), xz (y) and yz (x) planes.

        Args:
            data_dir (Path): Path to the directory for image output
            prefix (str): String to prepend to image filename
        """
        self.data_im_out_dir = data_dir
        logging.info("Slicing data volume and saving slices to disk")
        os.makedirs(data_dir, exist_ok=True)
        self._output_slices_to_disk(self.data_vol, data_dir, prefix)

    def output_label_slices(self, data_dir: Path, prefix: str) -> None:
        """
        Method that triggers slicing label data volume to disk in the
        xy (z), xz (y) and yz (x) planes.

        Args:
            data_dir (Path): Path to the directory for label image output
            prefix (str): String to prepend to image filename
        """
        self.seg_im_out_dir = data_dir
        logging.info("Slicing label volume and saving slices to disk")
        os.makedirs(data_dir, exist_ok=True)
        self._output_slices_to_disk(self.seg_vol, data_dir, prefix, label=True)

    def _output_slices_to_disk(self, data_arr, output_path, name_prefix, label=False):
        """Coordinates the slicing of an image volume in the three orthogonal
        planes to images on disk.

        Args:
            data_arr (array): The data volume to be sliced.
            output_path (pathlib.Path): A Path object to the output directory.
            name_prefix (str): Prefix for output filenames.
            label (bool): Whether this is a label volume.
        """
        # Labels are always processed in 2D mode
        if label or not self.use_2_5d_slicing:
            self._output_2d_slices_to_disk(data_arr, output_path, name_prefix, label)
        else:
            self._output_2_5d_slices_to_disk(data_arr, output_path, name_prefix)

    def _output_2d_slices_to_disk(self, data_arr, output_path, name_prefix, label=False):
        """Coordinates the slicing of an image volume in 2D mode (single channel).

        Args:
            data_arr (array): The data volume to be sliced.
            output_path (pathlib.Path): A Path object to the output directory.
            name_prefix (str): Prefix for output filenames.
            label (bool): Whether this is a label volume.
        """
        shape_tup = data_arr.shape
        axis_enum = utils.get_training_axis(self.settings)
        ax_idx_pairs = utils.get_axis_index_pairs(shape_tup, axis_enum)
        num_ims = utils.get_num_of_ims(shape_tup, axis_enum)
        for axis, index in tqdm(ax_idx_pairs, total=num_ims):
            out_path = output_path / f"{name_prefix}_{axis}_stack_{index}"
            self._output_im(
                utils.axis_index_to_slice(data_arr, axis, index), out_path, label
            )

    def _output_2_5d_slices_to_disk(self, data_arr, output_path, name_prefix):
        """Coordinates the slicing of an image volume in 2.5D mode (RGB channels).

        Args:
            data_arr (array): The data volume to be sliced.
            output_path (pathlib.Path): A Path object to the output directory.
            name_prefix (str): Prefix for output filenames.
        """
        shape_tup = data_arr.shape
        axis_enum = utils.get_training_axis(self.settings)
        ax_idx_pairs = utils.get_axis_index_pairs(shape_tup, axis_enum)
        num_ims = utils.get_num_of_ims(shape_tup, axis_enum)
        
        for axis, index in tqdm(ax_idx_pairs, total=num_ims):
            out_path = output_path / f"{name_prefix}_{axis}_stack_{index:06d}"
            rgb_slice = self._create_2_5d_slice(data_arr, axis, index)
            self._output_im(rgb_slice, out_path, label=False, is_rgb=True)

    def _create_2_5d_slice(self, data_arr, axis, index):
        """Creates a 2.5D RGB slice from adjacent slices along the specified axis.

        Args:
            data_arr (array): The data volume to be sliced.
            axis (str): The axis along which to slice ('z', 'y', 'x').
            index (int): The slice index.

        Returns:
            array: RGB image with shape (height, width, 3).
        """
        # Get the current slice
        current_slice = utils.axis_index_to_slice(data_arr, axis, index)
        
        # Get previous and next slices, handling border cases
        if axis == "z":
            depth = data_arr.shape[0]
            if index == 0:
                prev_slice = current_slice
                next_slice = utils.axis_index_to_slice(data_arr, axis, index + 1)
            elif index == depth - 1:
                prev_slice = utils.axis_index_to_slice(data_arr, axis, index - 1)
                next_slice = current_slice
            else:
                prev_slice = utils.axis_index_to_slice(data_arr, axis, index - 1)
                next_slice = utils.axis_index_to_slice(data_arr, axis, index + 1)
        elif axis == "y":
            depth = data_arr.shape[1]
            if index == 0:
                prev_slice = current_slice
                next_slice = utils.axis_index_to_slice(data_arr, axis, index + 1)
            elif index == depth - 1:
                prev_slice = utils.axis_index_to_slice(data_arr, axis, index - 1)
                next_slice = current_slice
            else:
                prev_slice = utils.axis_index_to_slice(data_arr, axis, index - 1)
                next_slice = utils.axis_index_to_slice(data_arr, axis, index + 1)
        elif axis == "x":
            depth = data_arr.shape[2]
            if index == 0:
                prev_slice = current_slice
                next_slice = utils.axis_index_to_slice(data_arr, axis, index + 1)
            elif index == depth - 1:
                prev_slice = utils.axis_index_to_slice(data_arr, axis, index - 1)
                next_slice = current_slice
            else:
                prev_slice = utils.axis_index_to_slice(data_arr, axis, index - 1)
                next_slice = utils.axis_index_to_slice(data_arr, axis, index + 1)
        
    
        
        rgb_image = np.stack([
            prev_slice,     # Red channel: previous slice
            current_slice,  # Green channel: current slice
            next_slice      # Blue channel: next slice
        ], axis=2)
        
        return rgb_image

    
    def _output_im(self, data, path, label=False, is_rgb=False):
        """Converts a slice of data into an image on disk.

        Args:
            data (numpy.array): The data slice to be converted.
            path (str): The path of the image file including the filename prefix.
            label (bool): Whether to convert values >1 to 1 for binary segmentation.
            is_rgb (bool): Whether the data is RGB (3 channels) or grayscale (1 channel).
        """
        if is_rgb:
            # RGB data is already normalized to 0-1, convert to uint8
            if data.dtype != np.uint8:
                data = (data * 255).astype(np.uint8)
        else:
            if data.dtype != np.uint8:
                data = img_as_ubyte(data)

        if label and not self.multilabel:
            data[data > 1] = 1
            
        io.imsave(f"{path}.png", data, check_contrast=False)

    def _delete_image_dir(self, im_dir_path):
        if im_dir_path.exists():
            ims = list(im_dir_path.glob("*.png"))
            logging.info(f"Deleting {len(ims)} images.")
            for im in ims:
                im.unlink()
            logging.info(f"Deleting the empty directory.")
            im_dir_path.rmdir()

    def clean_up_slices(self) -> None:
        """
        Deletes data and label image slices created by Slicer.
        """
        self._delete_image_dir(self.data_im_out_dir)
        self._delete_image_dir(self.seg_im_out_dir)
