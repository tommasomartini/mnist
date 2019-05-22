"""
This module contains all the constants used in this repo.

You should not need to modify this file (not too often, at least!).
"""
import os

import numpy as np

import mnist.custom_utils.readonly as ro


class Constants(ro.ReadOnly):
    # Extensions.
    IMAGE_EXTENSION = 'png'
    METADATA_EXTENSION = 'json'
    METAGRAPH_EXTENSION = 'meta'

    TRAINING_SET_NAME = 'training_set'
    VALIDATION_SET_NAME = 'validation_set'
    TEST_SET_NAME = 'test_set'

    LATEST_TRAINED_CKPT_NAME = 'latest_trained_ckpt'
    BEST_MODEL_CKPT_NAME = 'best_model_ckpt'

    MNIST_IMAGE_WIDTH = 28
    MNIST_IMAGE_HEIGTH = 28
    MNIST_IMAGE_CHANNELS = 1
    MNIST_NUM_CLASSES = 10

    IMAGE_DATA_TYPE = np.float32
    LABEL_DATA_TYPE = np.int32

    EPOCH_CURSOR_MULTIPLIER = 1e6

    # Training status file.
    TRAINING_STATUS_FILENAME = 'training_status'
    TRAINING_STATUS_EXTENSION = 'json'
    LATEST_TRAINED_KEY = 'latest_trained'
    LATEST_EVALUATED_KEY = 'latest_evaluated'
    BEST_KEY = 'best'
    EPOCH_IDX_KEY = 'epoch_idx'
    METRIC_KEY = 'mean_loss'
    DEFAULT_EPOCH_IDX = -1
    DEFAULT_METRIC_VALUE = np.inf


class MetadataFields(ro.ReadOnly):
    ID = 'id'
    LABEL = 'label'


# Should be something like: "dataset_split.json".
DatasetFilenames = {
    Constants.TRAINING_SET_NAME: os.path.extsep.join(
        (Constants.TRAINING_SET_NAME, Constants.METADATA_EXTENSION)),
    Constants.VALIDATION_SET_NAME: os.path.extsep.join(
        (Constants.VALIDATION_SET_NAME, Constants.METADATA_EXTENSION)),
    Constants.TEST_SET_NAME: os.path.extsep.join(
        (Constants.TEST_SET_NAME, Constants.METADATA_EXTENSION)),
}


class MetagraphFilenames(ro.ReadOnly):
    TRAINING = os.path.extsep.join(
        ('training_metagraph', Constants.METAGRAPH_EXTENSION))
    VALIDATION = os.path.extsep.join(
        ('validation_metagraph', Constants.METAGRAPH_EXTENSION))
    EVALUATION = os.path.extsep.join(
        ('evaluation_metagraph', Constants.METAGRAPH_EXTENSION))
