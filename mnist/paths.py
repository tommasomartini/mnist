import os

import mnist.config as config
import mnist.constants as constants
import mnist.custom_utils.readonly as ro

_CNST = constants.Constants
_dataset_def_dir = config.GeneralConfig.DATASET_DEF_DIR
_log_dir = config.GeneralConfig.LOG_DIR


class DatasetDefinitions(ro.ReadOnly):
    """Stores the paths to the files defining the datasets."""
    TRAINING = os.path.join(
        _dataset_def_dir,
        constants.DatasetFilenames[_CNST.TRAINING_SET_NAME]
    )
    VALIDATION = os.path.join(
        _dataset_def_dir,
        constants.DatasetFilenames[_CNST.VALIDATION_SET_NAME]
    )
    TEST = os.path.join(
        _dataset_def_dir,
        constants.DatasetFilenames[_CNST.TEST_SET_NAME]
    )


class MetaGraphs(ro.ReadOnly):
    """Stores the paths to the MetaGraph files, containing the definitions
    of different graphs."""
    TRAINING = os.path.join(_log_dir,
                            constants.MetagraphFilenames.TRAINING)
    VALIDATION = os.path.join(_log_dir,
                              constants.MetagraphFilenames.VALIDATION)


class Checkpoints(ro.ReadOnly):
    """Stores the paths to the checkpoints generated during an experiment."""
    LATEST_TRAINED = os.path.join(_log_dir,
                                  _CNST.LATEST_TRAINED_CKPT_NAME)


class TrainingStatus(ro.ReadOnly):
    PATH = os.path.join(
        _log_dir,
        os.path.extsep.join((_CNST.TRAINING_STATUS_FILENAME,
                             _CNST.TRAINING_STATUS_EXTENSION))
    )
