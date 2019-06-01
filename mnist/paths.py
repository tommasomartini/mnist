import os

import mnist.config as config
import mnist.constants as constants
import mnist.custom_utils.readonly as ro

_CNST = constants.Constants


def mkdir_if_not_exists(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


class BasePaths(ro.ReadOnly):
    # Where data are stored.
    DATA_DIR = os.environ[_CNST.DATA_DIR_ENVVAR]

    # Where the output of the experiments will be stored.
    BASE_LOG_DIR = os.environ[_CNST.BASE_LOG_DIR_ENVVAR]

    # Where the dataset definitions are stored.
    DATASET_DEF_DIR = os.environ[_CNST.DATASET_DEF_DIR_ENVVAR]

    # Log folder for this experiment.
    EXP_LOG_DIR = mkdir_if_not_exists(os.path.join(
        BASE_LOG_DIR,
        config.ExperimentConfig.EXPERIMENT_CODE))


class DatasetDefinitions(ro.ReadOnly):
    """Stores the paths to the files defining the datasets."""
    TRAINING = os.path.join(
        BasePaths.DATASET_DEF_DIR,
        constants.DatasetFilenames[_CNST.TRAINING_SET_NAME]
    )
    VALIDATION = os.path.join(
        BasePaths.DATASET_DEF_DIR,
        constants.DatasetFilenames[_CNST.VALIDATION_SET_NAME]
    )
    TEST = {
        test_set_name: os.path.join(
            BasePaths.DATASET_DEF_DIR,
            constants.DatasetFilenames[test_set_name])
        for test_set_name
        in _CNST.TEST_SET_NAMES
    }


class MetaGraphs(ro.ReadOnly):
    """Stores the paths to the MetaGraph files, containing the definitions
    of different graphs."""
    TRAINING = os.path.join(BasePaths.EXP_LOG_DIR,
                            constants.MetagraphFilenames.TRAINING)
    VALIDATION = os.path.join(BasePaths.EXP_LOG_DIR,
                              constants.MetagraphFilenames.VALIDATION)
    EVALUATION = os.path.join(BasePaths.EXP_LOG_DIR,
                              constants.MetagraphFilenames.EVALUATION)
    INFERENCE = os.path.join(BasePaths.EXP_LOG_DIR,
                             constants.MetagraphFilenames.INFERENCE)


class Checkpoints(ro.ReadOnly):
    """Stores the paths to the checkpoints generated during an experiment."""
    LATEST_TRAINED = os.path.join(BasePaths.EXP_LOG_DIR,
                                  _CNST.LATEST_TRAINED_CKPT_NAME)
    BEST_MODEL = os.path.join(BasePaths.EXP_LOG_DIR, _CNST.BEST_MODEL_CKPT_NAME)


class TrainingStatus(ro.ReadOnly):
    PATH = os.path.join(
        BasePaths.EXP_LOG_DIR,
        os.path.extsep.join((_CNST.TRAINING_STATUS_FILENAME,
                             _CNST.TRAINING_STATUS_EXTENSION))
    )
