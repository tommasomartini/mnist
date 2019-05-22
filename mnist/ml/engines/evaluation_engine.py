import glob
import os
from itertools import count

import tensorflow as tf

import mnist.config as config
import mnist.ml.model.graphs as graphs
import mnist.ml.model.naming as naming
import mnist.ml.training_utils as utils
import mnist.paths as paths
from mnist.custom_utils.logger import std_logger as logger


class EvaluationEngine:

    def __init__(self):
        # Create and store the MetaGraph for the evaluation graph.
        metagraph_path = paths.MetaGraphs.EVALUATION
        if os.path.exists(metagraph_path):
            logger.info('Importing existing evaluation MetaGraph '
                        'from {}'.format(metagraph_path))
            evaluation_graph = tf.Graph()
            with evaluation_graph.as_default():
                tf.train.import_meta_graph(metagraph_path)
        else:
            logger.info('Creating new evaluation MetaGraph')
            evaluation_graph = graphs.build_evaluation_graph()
            with evaluation_graph.as_default():
                tf.train.export_meta_graph(metagraph_path)

        self._session = tf.Session(graph=evaluation_graph)

        with evaluation_graph.as_default():
            self._saver = tf.train.Saver()

    def shut_down(self):
        """Shuts down the tf.Session associated with the validation."""
        self._session.close()

    def evaluate_best_model_on_dataset(self, dataset_def):
        dataset_size = len(dataset_def)
        batches_per_epoch = utils.batches_per_epoch(
            dataset_size=dataset_size,
            batch_size=config.TrainingConfig.BATCH_SIZE_TEST,
            drop_last=False)

        best_ckpt_prefix = paths.Checkpoints.BEST_MODEL
        best_ckpt_pattern = best_ckpt_prefix + '.*'
        if not glob.glob(best_ckpt_pattern):
            raise IOError('No checkpoint at {}'.format(best_ckpt_prefix))

        # Load the best trained weights.
        self._saver.restore(self._session, best_ckpt_prefix)

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
                if batch_idx > batches_per_epoch:
                    logger.warning('Batch index is {} but an epoch should '
                                   'only contain '
                                   '{} batches'.format(batch_idx,
                                                       batches_per_epoch))
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
