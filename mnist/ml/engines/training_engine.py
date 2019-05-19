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


class TrainingEngine:

    def __init__(self):
        # Load the training set definition. It will be used to know the dataset
        # size and possibly to initialize a new training graph.
        training_set_def_path = paths.DatasetDefinitions.TRAINING
        with open(training_set_def_path, 'r') as f:
            training_set_def = json.load(f)

        self.dataset_size = len(training_set_def)
        self.batches_per_epoch = utils.batches_per_epoch(
            dataset_size=self.dataset_size,
            batch_size=config.TrainingConfig.BATCH_SIZE_TRAINING,
            drop_last=True)

        # Create and save the MetaGraph for the training graph.
        metagraph_path = paths.MetaGraphs.TRAINING
        if os.path.exists(metagraph_path):
            logger.info('Importing existing training MetaGraph '
                        'from {}'.format(metagraph_path))
            training_graph = tf.Graph()
            with training_graph.as_default():
                tf.train.import_meta_graph(metagraph_path)
        else:
            logger.info('Creating new training MetaGraph')
            training_graph = graphs.build_training_graph(training_set_def)
            with training_graph.as_default():
                tf.train.export_meta_graph(metagraph_path)

        self._session = tf.Session(graph=training_graph)

        with training_graph.as_default():
            self._saver = tf.train.Saver(max_to_keep=1)

    def shut_down(self):
        """Shuts down the tf.Session associated with the training."""
        self._session.close()

    def initialize(self):
        """Performs all the operations needed to initialize
        the training session."""
        # Fetch and run the initialization operations.
        with self._session.graph.as_default():
            init_op = tf.global_variables_initializer()
            dataset_init_op = self._session.graph.get_operation_by_name(
                naming.Names.DATASET_INIT_OP)
        self._session.run(init_op)
        self._session.run(dataset_init_op)

    def train_epoch(self):
        """Performs the training of one epoch of the training set."""
        train_op = self._session.graph.get_operation_by_name(
            naming.Names.TRAINING_OPERATION)
        train_loss = self._session.graph.get_collection(tf.GraphKeys.LOSSES)[0]
        loss_summary = self._session.graph.get_collection(
            naming.Names.TRAINING_SUMMARY_COLLECTION)[0]

        for batch_idx in count():
            if batch_idx >= self.batches_per_epoch:
                logger.warning('Batch index is {} but an epoch should '
                               'only contain '
                               '{} batches'.format(batch_idx,
                                                   self.batches_per_epoch))
            try:
                (_train_op_out,
                 train_loss_out,
                 loss_summary_out) = self._session.run([train_op,
                                                        train_loss,
                                                        loss_summary])
            except tf.errors.OutOfRangeError:
                break
            yield batch_idx, train_loss_out, loss_summary_out

    def resume(self):
        """Resumes the training session from a checkpoint file."""
        self._saver.restore(self._session, paths.Checkpoints.LATEST_TRAINED)

    def save(self):
        """Saves on disk the checkpoint of the current session status.

        Only the trained weights are stored, not the MetaGraph defining the
        network architecture. The MetaGraph is stored in a separate file.
        """
        self._saver.save(self._session,
                         paths.Checkpoints.LATEST_TRAINED,
                         write_meta_graph=False)
