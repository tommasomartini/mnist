"""
This module contains all the constants used in this repo.

You should not need to modify this file (not too often, at least!).
"""
import os

import mnist.custom_utils.readonly as ro


class Constants(ro.ReadOnly):
    IMAGE_EXTENSION = 'png'
    METADATA_EXTENSION = 'json'
    TRAINING_SET_NAME = 'training_set'
    VALIDATION_SET_NAME = 'validation_set'
    TEST_SET_NAME = 'test_set'


class MetadataFields(ro.ReadOnly):
    ID = 'id'
    LABEL = 'label'


class Filenames(ro.ReadOnly):
    # Should be something like: "dataset_spit.json".
    TRAINING_SET = os.path.extsep.join((Constants.TRAINING_SET_NAME,
                                        Constants.METADATA_EXTENSION))
    VALIDATION_SET = os.path.extsep.join((Constants.VALIDATION_SET_NAME,
                                          Constants.METADATA_EXTENSION))
    TEST_SET = os.path.extsep.join((Constants.TEST_SET_NAME,
                                    Constants.METADATA_EXTENSION))


class LoggerNames(ro.ReadOnly):
    STD = 'STD'
    DATA_SETUP = 'DATA_SETUP'
