import glob
import json
import os
from itertools import count

import tensorflow as tf

import mnist.config as config
import mnist.ml.model.graphs as graphs
import mnist.ml.model.naming as naming
import mnist.ml.training_utils as utils
import mnist.paths as paths
from mnist.custom_utils.logger import std_logger as logger


class ValidationEngine:

    def __init__(self):
        # Load the validation set definition. It will be used to know
        # the dataset size and possibly to initialize a new validation graph.
        validation_set_def_path = paths.DatasetDefinitions.VALIDATION
        with open(validation_set_def_path, 'r') as f:
            validation_set_def = json.load(f)

        self.dataset_size = len(validation_set_def)
        self.batches_per_epoch = utils.batches_per_epoch(
            dataset_size=self.dataset_size,
            batch_size=config.TrainingConfig.BATCH_SIZE_TRAINING,
            drop_last=False)

        # Create and save the MetaGraph for the validation graph.
        metagraph_path = paths.MetaGraphs.VALIDATION
        if os.path.exists(metagraph_path):
            logger.info('Importing existing validation MetaGraph '
                        'from {}'.format(metagraph_path))
            validation_graph = tf.Graph()
            with validation_graph.as_default():
                tf.train.import_meta_graph(metagraph_path)
        else:
            logger.info('Creating new validation MetaGraph')
            validation_graph = graphs.build_validation_graph(validation_set_def)
            with validation_graph.as_default():
                tf.train.export_meta_graph(metagraph_path)

        self._session = tf.Session(graph=validation_graph)

        with validation_graph.as_default():
            self._saver = tf.train.Saver(max_to_keep=1)

    def shut_down(self):
        """Shuts down the tf.Session associated with the validation."""
        self._session.close()

    def save(self):
        """Saves on disk the checkpoint of the current session status.

        Only the trained weights are stored, not the MetaGraph defining the
        network architecture. The MetaGraph is stored in a separate file.
        """
        self._saver.save(self._session,
                         paths.Checkpoints.BEST_MODEL,
                         write_meta_graph=True)

    def evaluate_latest_trained_model(self):
        """Evaluates the latest trained model on the validation set."""
        latest_ckpt_prefix = paths.Checkpoints.LATEST_TRAINED
        latest_ckpt_pattern = latest_ckpt_prefix + '.*'
        if not glob.glob(latest_ckpt_pattern):
            raise IOError('No checkpoint at {}'.format(latest_ckpt_prefix))

        # Load the latest trained weights.
        self._saver.restore(self._session, latest_ckpt_prefix)

        # Fetch the output nodes.
        loss = self._session.graph.get_tensor_by_name(
            naming.Names.EVALUATION_LOSS + ':0')
        (_probabilities,
         _prediction,
         num_correct_predictions,
         batch_size) = self._session.graph.get_collection(
            naming.Names.OUTPUT_COLLECTION)

        with self._session.graph.as_default():
            # Initialize the validation set.
            dataset_init_op = self._session.graph.get_operation_by_name(
                naming.Names.DATASET_INIT_OP)
            self._session.run(dataset_init_op)

            for batch_idx in count():
                if batch_idx > self.batches_per_epoch:
                    logger.warning('Batch index is {} but an epoch should '
                                   'only contain '
                                   '{} batches'.format(batch_idx,
                                                       self.batches_per_epoch))
                try:
                    (loss_out,
                     true_positives,
                     batch_size_out) = self._session.run([
                        loss,
                        num_correct_predictions,
                        batch_size,
                    ])
                except tf.errors.OutOfRangeError:
                    break

                yield batch_idx, loss_out, true_positives, batch_size_out
