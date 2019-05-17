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

class TrainingConfig(ro.ReadOnly):
    """Container for training configurations."""
    NUM_EPOCHS = 50
    BATCH_SIZE_TRAINING = 100
    BATCH_SIZE_VALIDATION = BATCH_SIZE_TRAINING
    BATCH_SIZE_TEST = BATCH_SIZE_TRAINING
    LEARNING_RATE = 1e-3
    DROP_LAST_INCOMPLETE_BATCH = True
