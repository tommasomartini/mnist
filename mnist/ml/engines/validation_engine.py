import json
import os
from itertools import count

import tensorflow as tf

import mnist.ml.model.graphs as graphs
import mnist.ml.model.naming as naming
import mnist.paths as paths
from mnist.custom_utils.logger import std_logger as logger


class ValidationEngine:

    def __init__(self):

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
            validation_set_def_path = paths.DatasetDefinitions.VALIDATION
            with open(validation_set_def_path, 'r') as f:
                validation_set_def = json.load(f)

            validation_graph = graphs.build_validation_graph(validation_set_def)
            with validation_graph.as_default():
                tf.train.export_meta_graph(metagraph_path)

        self._session = tf.Session(graph=validation_graph)

        with validation_graph.as_default():
            self._saver = tf.train.Saver(max_to_keep=1)

    def shut_down(self):
        """Shuts down the tf.Session associated with the validation."""
        self._session.close()

    def initialize(self):
        """Performs all the operations needed to initialize
        the validation session."""
        # Fetch and run the initialization operations.
        with self._session.graph.as_default():
            init_op = tf.global_variables_initializer()
            dataset_init_op = self._session.graph.get_operation_by_name(
                naming.Names.DATASET_INIT_OP)
        self._session.run(init_op)
        self._session.run(dataset_init_op)

    def evaluate(self):
        """Performs the evaluation of one epoch of the validation set."""
        loss = self._session.graph.get_tensor_by_name(
            naming.Names.EVALUATION_LOSS + ':0')

        _probabilities, \
        _prediction, \
        num_correct_predictions, \
        batch_size = self._session.graph.get_collection(
            naming.Names.OUTPUT_COLLECTION)

        with self._session.graph.as_default():
            for batch_idx in count():
                try:
                    loss_out, \
                    true_positives, \
                    batch_size_out = self._session.run([
                        loss,
                        num_correct_predictions,
                        batch_size,
                    ])
                except tf.errors.OutOfRangeError:
                    break

                yield batch_idx, loss_out, true_positives, batch_size_out
