"""
This module contains all the constants used in this repo.

You should not need to modify this file (not too often, at least!).
"""
import mnist.custom_utils.readonly as ro


class Constants(ro.ReadOnly):
    IMAGE_EXTENSION = 'png'
    METADATA_EXTENSION = 'json'


class MetadataFields(ro.ReadOnly):
    ID = 'id'
    LABEL = 'label'


class Filenames(ro.ReadOnly):
    TRAINING_SET = 'training_set.json'
    VALIDATION_SET = 'validation_set.json'
    TEST_SET = 'test_set.json'


class LoggerNames(ro.ReadOnly):
    STD = 'STD'
    DATA_SETUP = 'DATA_SETUP'
