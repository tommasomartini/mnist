import json
import os

from tqdm import tqdm

import mnist.config as config
import mnist.constants as constants
import mnist.ml.engines.evaluation_engine as ev_eng
import mnist.ml.engines.inference_engine as inf_eng
import mnist.ml.engines.logging_engine as log_eng
import mnist.ml.engines.training_engine as tr_eng
import mnist.ml.engines.validation_engine as val_eng
import mnist.ml.evaluation_utils as eval_utils
import mnist.ml.training_utils as train_utils
import mnist.paths as paths
from mnist.custom_utils.logger import DISABLE_PROGRESS_BAR
from mnist.custom_utils.logger import PROGRESS_BAR_PREFIX
from mnist.custom_utils.logger import std_logger as logger

_CNST = constants.Constants


def _load_or_create_training_status(training_status_path):
    """Returns a dict containing the current status of the experiment.

    This function tries to load the current status from disk, in case an
    existing one has been saved; otherwise returns a new one.

    Args:
        training_status_path (str): Path to the training status file to load.

    Returns:
        A dict containing the current experiment status.
    """
    if os.path.exists(training_status_path):
        # Load an existing training status.
        with open(training_status_path, 'r') as f:
            training_status = json.load(f)
        logger.info('Loaded existing training status '
                    'from {}'.format(training_status_path))
    else:
        # Create a new training status.
        training_status = {
            _CNST.LATEST_TRAINED_KEY: _CNST.DEFAULT_EPOCH_IDX,
            _CNST.LATEST_EVALUATED_KEY: {
                _CNST.EPOCH_IDX_KEY: _CNST.DEFAULT_EPOCH_IDX,
                _CNST.METRIC_KEY: _CNST.DEFAULT_METRIC_VALUE,
            },
            _CNST.BEST_KEY: {
                _CNST.EPOCH_IDX_KEY: _CNST.DEFAULT_EPOCH_IDX,
                _CNST.METRIC_KEY: _CNST.DEFAULT_METRIC_VALUE,
            },
        }
        logger.info('Created new training status')

    return training_status


class ExperimentScheduler:

    def __init__(self, log_dir):
        if not os.path.isdir(log_dir):
            raise IOError('Invalid log folder: {}'.format(log_dir))
        self._log_dir = log_dir

        # Initialize the training status.
        self._training_status_path = paths.TrainingStatus.PATH
        self._training_status = \
            _load_or_create_training_status(self._training_status_path)

        # Create the engines. In this phase all the MetaGraphs are also stored
        # on disk for later re-use.
        self._training_engine = tr_eng.TrainingEngine()
        self._validation_engine = val_eng.ValidationEngine()
        self._evaluation_engine = ev_eng.EvaluationEngine()
        self._logging_engine = log_eng.LoggingEngine(
            log_dir=self._log_dir,
            graph=self._training_engine.graph)
        inf_eng.InferenceEngine().create_metagraph()

    ############################################################################
    # Private methods ##########################################################
    ############################################################################

    def _shut_down_engines(self):
        self._training_engine.shut_down()
        self._validation_engine.shut_down()
        self._evaluation_engine.shut_down()
        self._logging_engine.shut_down()

    def _save_training_status(self):
        with open(self._training_status_path, 'w') as f:
            json.dump(self._training_status, f, indent=4)

    def _update_training_status(self, epoch_idx, validation_loss):
        is_new_best_epoch = False
        latest_evaluated_epoch = {
            _CNST.EPOCH_IDX_KEY: epoch_idx,
            _CNST.METRIC_KEY: validation_loss,
        }
        best_epoch = self._training_status[_CNST.BEST_KEY]
        if validation_loss < best_epoch[_CNST.METRIC_KEY]:
            best_epoch = latest_evaluated_epoch
            is_new_best_epoch = True
        self._training_status = {
            _CNST.LATEST_TRAINED_KEY: epoch_idx,
            _CNST.LATEST_EVALUATED_KEY: latest_evaluated_epoch,
            _CNST.BEST_KEY: best_epoch,
        }
        self._save_training_status()
        return is_new_best_epoch

    def _run_validation(self, epoch_idx):
        """Evaluates the latest trained model on the validation set.

        Args:
            epoch_idx (int): 0-based index of the latest trained epoch.

        Returns:
            A tuple (<average loss>, <accuracy>).
        """
        pbar_desc = '{} Evaluating after {} training epochs' \
                    ''.format(PROGRESS_BAR_PREFIX, epoch_idx + 1)
        pbar = tqdm(
            self._validation_engine.evaluate_latest_trained_model(),
            desc=pbar_desc,
            total=self._validation_engine.batches_per_epoch,
            leave=False,
            disable=DISABLE_PROGRESS_BAR)
        avg_loss, accuracy = eval_utils.evaluation_accumulator(pbar)
        return avg_loss, accuracy

    def _run_evaluation(self, testset_def_path, testset_name=None):
        """Evaluates the best trained model on a test set.

        Args:
            testset_def_path (str): Path to the dataset definition.
            testset_name (str, optional): Identifier associated to the current
                dataset. Defaults to None.

        Returns:
            A tuple (<average loss>, <accuracy>).
        """
        avg_loss, accuracy = eval_utils.run_evaluation(
            evaluation_engine=self._evaluation_engine,
            dataset_def_path=testset_def_path,
            dataset_name=testset_name
        )
        return avg_loss, accuracy

    ############################################################################
    # Scheduling functions #####################################################
    ############################################################################

    def _before_new_training(self):
        # Randomly initialize the training graph and evaluate the un-trained
        # model to set a worst-case baseline.
        self._training_engine.initialize()
        self._training_engine.save()
        epoch_idx = -1
        avg_loss, accuracy = self._run_validation(epoch_idx)
        logger.info('Untrained model: '
                    'loss={:.3f}, '
                    'accuracy={:.3f}'.format(avg_loss, accuracy))

        # Log the evaluation results.
        self._logging_engine.log_evaluation_results(epoch_idx=epoch_idx,
                                                    avg_loss=avg_loss,
                                                    accuracy=accuracy)

        # Update the training status.
        self._update_training_status(epoch_idx=epoch_idx,
                                     validation_loss=avg_loss)

    def _after_training(self):
        for testset_name, testset_def_path \
                in paths.DatasetDefinitions.TEST.items():
            avg_loss, accuracy = self._run_evaluation(testset_def_path,
                                                      testset_name)
            logger.info('Evaluated on: {}\n'
                        '  loss: {:.3f}\n'
                        '  accuracy: {:.3f}'.format(testset_name,
                                                    avg_loss,
                                                    accuracy))

    def _before_epoch(self, epoch_idx):
        pass

    def _after_epoch(self, epoch_idx):
        # Evaluate on the validation set.
        avg_loss, accuracy = self._run_validation(epoch_idx)
        # Log the evaluation results.
        self._logging_engine.log_evaluation_results(epoch_idx=epoch_idx,
                                                    avg_loss=avg_loss,
                                                    accuracy=accuracy)

        log_msg = 'After {} training epochs: ' \
                  'loss={:.3f}, ' \
                  'accuracy={:.3f}'.format(epoch_idx + 1,
                                           avg_loss,
                                           accuracy)
        is_new_best_epoch = \
            self._update_training_status(epoch_idx=epoch_idx,
                                         validation_loss=avg_loss)
        if is_new_best_epoch:
            log_msg += ' [new best model]'
            self._validation_engine.save()
        logger.info(log_msg)

    def _run_epoch(self, epoch_idx):
        pbar_desc = '{} Training epoch {}/{}'.format(
            PROGRESS_BAR_PREFIX,
            epoch_idx + 1,
            config.ExperimentConfig.NUM_EPOCHS)
        pbar = tqdm(self._training_engine.train_epoch(),
                    total=self._training_engine.batches_per_epoch,
                    desc=pbar_desc,
                    disable=DISABLE_PROGRESS_BAR)
        for batch_idx, loss, train_summary in pbar:
            cursor = train_utils.epoch_cursor(
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                batches_per_epoch=self._training_engine.batches_per_epoch)
            self._logging_engine.log_summary(summary=train_summary,
                                             epoch_cursor=cursor)

        # Save the checkpoint on disk.
        self._training_engine.save()

        # Update the training status.
        self._training_status[_CNST.LATEST_TRAINED_KEY] = epoch_idx
        self._save_training_status()

    ############################################################################
    # Public interface #########################################################
    ############################################################################

    def run(self):
        num_epochs = config.ExperimentConfig.NUM_EPOCHS

        latest_trained_epoch_idx = \
            self._training_status[_CNST.LATEST_TRAINED_KEY]

        if latest_trained_epoch_idx == -1:
            # No epoch has been trained: start a new training.
            self._before_new_training()
        else:
            # Some epochs were trained. Was the last epoch evaluated?
            latest_evaluated = self._training_status[_CNST.LATEST_EVALUATED_KEY]
            latest_evaluated_epoch_idx = latest_evaluated[_CNST.EPOCH_IDX_KEY]
            if latest_evaluated_epoch_idx == latest_trained_epoch_idx - 1:
                # The latest trained epoch was not evaluated.
                self._after_epoch(latest_trained_epoch_idx)
            elif latest_evaluated_epoch_idx < latest_trained_epoch_idx - 1:
                raise ValueError('Missing evaluations: the latest trained '
                                 'epoch index is {}, but the latest evaluated '
                                 'epoch index is {}'
                                 ''.format(latest_trained_epoch_idx,
                                           latest_evaluated_epoch_idx))
            else:
                # The latest trained epoch was evaluated: do nothing.
                pass

            # Resume the training where it was left.
            self._training_engine.resume()

        # At this point every trained epoch (possibly none) has been evaluated.

        for epoch_idx in range(latest_trained_epoch_idx + 1, num_epochs):
            self._before_epoch(epoch_idx)
            self._run_epoch(epoch_idx)
            self._after_epoch(epoch_idx)

        logger.info('Training completed')

        self._after_training()
        self._shut_down_engines()
