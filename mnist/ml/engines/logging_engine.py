import os

import tensorflow as tf

import mnist.ml.model.naming as naming
from mnist.ml.model.graphs import build_logging_graph
import mnist.ml.training_utils as training_utils


class LoggingEngine(object):

    def __init__(self, log_dir):
        if not os.path.isdir(log_dir):
            raise IOError('Invalid log folder: {}'.format(log_dir))

        self._summary_writer = tf.summary.FileWriter(log_dir, graph=None)

        plot_graph = build_logging_graph()
        self._session = tf.Session(graph=plot_graph)

    def shut_down(self):
        """Shuts down the tf.Session and the SummaryWriters."""
        self._summary_writer.close()
        self._session.close()

    def log_summary(self, summary, epoch_cursor):
        self._summary_writer.add_summary(summary, epoch_cursor)

    def log_evaluation_results(self,
                               epoch_idx,
                               avg_batch_loss,
                               accuracy):
        with self._session.graph.as_default():
            avg_batch_loss_placeholder = \
                self._session.graph.get_tensor_by_name(
                    naming.Names.AVG_BATCH_LOSS_PLACEHOLDER + ':0')
            accuracy_placeholder = \
                self._session.graph.get_tensor_by_name(
                    naming.Names.ACCURACY_PLACEHOLDER + ':0')

            avg_batch_loss_summary, \
            accuracy_summary = \
                tf.get_collection(naming.Names.EVALUATION_SUMMARY_COLLECTION)

            avg_batch_loss_summary_out = self._session.run(
                avg_batch_loss_summary,
                feed_dict={avg_batch_loss_placeholder: avg_batch_loss})
            accuracy_summary_out = self._session.run(
                accuracy_summary,
                feed_dict={accuracy_placeholder: accuracy})

        cursor_latest_epoch = training_utils.end_of_epoch_cursor(epoch_idx)

        self._summary_writer.add_summary(avg_batch_loss_summary_out, cursor_latest_epoch)
        self._summary_writer.add_summary(accuracy_summary_out, cursor_latest_epoch)
        self._summary_writer.flush()
