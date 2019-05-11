"""
This module contains all the settings and configurations.
"""
import logging

import mnist.constants as constants
import mnist.custom_utils.readonly as ro


class GeneralConfig(ro.ReadOnly):
    """Container for general configurations."""
    RANDOM_SEED = 0
    DATA_DIR = ''
    NUM_HASH_SUBDIR_LEVELS = 2
    LOGGING_LEVEL = logging.INFO

    # Datasets.
    DATASET_DEF_DIR = ''
    DATA_SPLIT_WEIGHTS = {
        constants.Constants.TRAINING_SET_NAME: 0.5,
        constants.Constants.VALIDATION_SET_NAME: 0.25,
        constants.Constants.TEST_SET_NAME: 0.25,
    }
