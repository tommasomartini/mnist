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

    TRAINING_CKPT_NAME = 'training_ckpt'

    MNIST_IMAGE_WIDTH = 28
    MNIST_IMAGE_HEIGTH = 28
    MNIST_IMAGE_CHANNELS = 1
    MNIST_NUM_CLASSES = 10

    IMAGE_DATA_TYPE = np.float32
    LABEL_DATA_TYPE = np.int32

    EPOCH_CURSOR_MULTIPLIER = 1e6


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
