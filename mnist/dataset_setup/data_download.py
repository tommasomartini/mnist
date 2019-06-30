import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

import mnist.file_interface as fi
import mnist.paths as paths
from mnist.constants import MetadataFields
from mnist.custom_utils.logger import PROGRESS_BAR_PREFIX
from mnist.custom_utils.logger import std_logger as logger


def _save_sample_image(data_dir, sample_id, raw_image, dry=False):
    """Stores a sample image on disk.

    Args:
        data_dir (str): Path to the base data folder.
        sample_id (str): Identifies the sample.
        raw_image (:obj:`numpy.array`): Raw image to save.
        dry (bool, optional): If True, no change is applied to the fie system.
            Defaults to False.
    """
    image_path = fi.get_image_path(data_dir, sample_id)
    image_dir = os.path.dirname(image_path)
    pil_img = Image.fromarray(raw_image).convert('L')
    if not dry:
        paths.mkdir_if_not_exists(image_dir)
        pil_img.save(image_path)


def _save_sample_metadata(data_dir, sample_id, label, dry=False):
    """Stores a sample metadata file on disk.

    Args:
        data_dir (str): Path to the base data folder.
        sample_id (str): Identifies the sample.
        label (int): Class index of the sample.
        dry (bool, optional): If True, no change is applied to the fie system.
            Defaults to False.
    """
    meta_path = fi.get_metadata_path(data_dir, sample_id)
    meta_dir = os.path.dirname(meta_path)
    meta = {
        MetadataFields.ID: sample_id,
        MetadataFields.LABEL: label,
    }
    if not dry:
        paths.mkdir_if_not_exists(meta_dir)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)


def download_mnist_data(data_dir, silent=False, dry=False):
    """Downloads the MNIST dataset and stores it on disk.

    The MNIST dataset is downloaded from the Keras repository:
        https://keras.io/datasets/#mnist-database-of-handwritten-digits

    Args:
        data_dir (str): Path to the base data folder.
        silent (bool, optional): If True, no progress bar is shown.
            Defaults to False.
        dry (bool, optional): If True, no change is applied to the fie system.
            Defaults to False.
    """
    if dry:
        logger.warning('Dry run!')

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    raw_images = np.concatenate([x_train, x_test])  # shape (70000, 28, 28)
    raw_labels = np.concatenate([y_train, y_test])  # shape (70000,)

    sample_idx = -1

    pbar = tqdm(enumerate(zip(raw_images,
                              raw_labels)),
                desc='{} Downloading data'.format(PROGRESS_BAR_PREFIX),
                total=len(raw_images),
                disable=silent)
    for sample_idx, (image, label) in pbar:
        sample_id = str(sample_idx)
        _save_sample_image(data_dir=data_dir,
                           sample_id=sample_id,
                           raw_image=image,
                           dry=dry)
        _save_sample_metadata(data_dir=data_dir,
                              sample_id=sample_id,
                              label=int(label),
                              dry=dry)

    log_msg = 'Downloaded {} samples in {}'.format(sample_idx + 1, data_dir)
    if dry:
        log_msg = '(Dry-)' + log_msg
    logger.info(log_msg)
