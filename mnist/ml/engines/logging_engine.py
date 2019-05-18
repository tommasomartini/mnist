import os

import tensorflow as tf

from mnist.ml.model.graphs import build_plotting_graph


class LoggingEngine(object):

    def __init__(self, log_dir):
        if not os.path.isdir(log_dir):
            raise IOError('Invalid log folder: {}'.format(log_dir))

        self._summary_writer = tf.summary.FileWriter(log_dir, graph=None)

        plot_graph = build_plotting_graph()
        self._plotting_session = tf.Session(graph=plot_graph)

    def shut_down(self):
        """Shuts down the tf.Session and the SummaryWriters."""
        self._summary_writer.close()
        self._plotting_session.close()

    def log_summary(self, summary, epoch_cursor):
        self._summary_writer.add_summary(summary, epoch_cursor)
