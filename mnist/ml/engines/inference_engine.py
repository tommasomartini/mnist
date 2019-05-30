import glob
import os

import tensorflow as tf

import mnist.data.preprocessing.image_preprocessing as img_preproc
import mnist.ml.model.graphs as graphs
import mnist.ml.model.naming as naming
import mnist.paths as paths
from mnist.custom_utils.logger import std_logger as logger


class InferenceEngine:

    def __init__(self):
        self._session = None

    def create_metagraph(self):
        # Creates and stores the MetaGraph for the inference graph.
        metagraph_path = paths.MetaGraphs.INFERENCE
        if not os.path.exists(metagraph_path):
            logger.info('Creating new inference MetaGraph')
            inference_graph = graphs.build_inference_graph()
            with inference_graph.as_default():
                tf.train.export_meta_graph(metagraph_path)

    def shut_down(self):
        """Shuts down the tf.Session associated with the inference."""
        if self._session is not None:
            self._session.close()

    def load(self):
        metagraph_path = paths.MetaGraphs.INFERENCE
        if not os.path.exists(metagraph_path):
            raise IOError('No MetaGraph at {}'.format(metagraph_path))

        # Load the MetaGraph from disk.
        inference_graph = tf.Graph()
        with inference_graph.as_default():
            tf.train.import_meta_graph(metagraph_path)
            self._saver = tf.train.Saver()

        logger.info('Imported existing inference MetaGraph '
                    'from {}'.format(metagraph_path))

        # Create a tf.Session object with the loaded graph.
        self._session = tf.Session(graph=inference_graph)

        # Load the trained weights.
        best_ckpt_prefix = paths.Checkpoints.BEST_MODEL
        best_ckpt_pattern = best_ckpt_prefix + '.*'
        if not glob.glob(best_ckpt_pattern):
            raise IOError('No checkpoint at {}'.format(best_ckpt_prefix))

        self._saver.restore(self._session, best_ckpt_prefix)

        logger.info('Loaded trained weights '
                    'from {}'.format(best_ckpt_prefix))

    def load_preprocess_and_predict(self, image_paths):
        if self._session is None:
            raise ValueError('This {} instance was not loaded'
                             ''.format(self.__class__.__name__))

        # Fetch the input nodes.
        image_ph = self._session.graph.get_tensor_by_name(
            naming.Names.IMAGES_PLACEHOLDER + ':0')

        # Fetch the output nodes.
        prediction, probabilities = self._session.graph.get_collection(
            naming.Names.INFERENCE_OUTPUT_COLLECTION)

        with self._session.graph.as_default():
            images = self._session.run([
                img_preproc.load_and_preprocess_image(image_path)
                for image_path in image_paths
            ])

            prediction_out, probabilities_out = self._session.run(
                [prediction, probabilities],
                feed_dict={image_ph: images}
            )

        return prediction_out, probabilities_out

    def __enter__(self):
        if not self._session:
            self.load()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shut_down()
