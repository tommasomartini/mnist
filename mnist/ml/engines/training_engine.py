import json
import os

import tensorflow as tf

import mnist.ml.model.graphs as graphs
import mnist.paths as paths
from mnist.custom_utils.logger import std_logger as logger


class TrainingEngine:

    def __init__(self):

        # Create and save the MetaGraph for the training graph.
        if os.path.exists(paths.MetaGraphs.TRAINING):
            logger.info('Importing existing training MetaGraph '
                        'from {}'.format(paths.MetaGraphs.TRAINING))
            training_graph = tf.Graph()
            with training_graph.as_default():
                tf.train.import_meta_graph(paths.MetaGraphs.TRAINING)
        else:
            logger.info('Creating new training MetaGraph')
            training_dataset_def_path = paths.DatasetDefinitions.TRAINING
            with open(training_dataset_def_path, 'r') as f:
                training_dataset_def = json.load(f)

            training_graph = graphs.build_training_graph(training_dataset_def)
            with training_graph.as_default():
                tf.train.export_meta_graph(paths.MetaGraphs.TRAINING)

        self._session = tf.Session(graph=training_graph)

        with training_graph.as_default():
            self._training_saver = tf.train.Saver(max_to_keep=1)

    def close(self):
        """Shuts down the tf.Session associated with the training."""
        self._session.close()

    def initialize(self):
        """Performs all the operations needed to initialize
        the training session."""
        with self._session.graph.as_default():
            init_op = tf.global_variables_initializer()
        self._session.run(init_op)

    def train_epoch(self):
        """Performs the training of one epoch of the training set."""
        raise NotImplementedError()
        # for training_tuple in training_iterable_on_dataset(self._session):
        #     yield training_tuple

    def resume(self):
        """Resumes the training session from a checkpoint file."""
        self._training_saver.restore(self._session, paths.Checkpoints.TRAINING)

    def save(self):
        """Saves on disk the checkpoint of the current session status.

        Only the trained weights are stored, not the MetaGraph defining the
        network architecture. The MetaGraph is stored in a separate file.
        """
        self._training_saver.save(self._session,
                                  paths.Checkpoints.TRAINING,
                                  write_meta_graph=False)
