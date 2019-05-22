import logging
import os

import tensorflow as tf
from python_log_indenter import IndentedLoggerAdapter

import mnist.config as config

# General logger.
_LOGGING_LEVEL = config.GeneralConfig.LOGGING_LEVEL
_logging_format = '[%(levelname)s] %(message)s'

# Warning!
# When using TensorFlow, the standard logging is shadowed by absl.logging, as
# described here:
#   https://www.tensorflow.org/alpha/guide/effective_tf2#api_cleanup
#
# The logger obtained by logging.get_logger() has a single element: an
# absl.logging.ABSLHandler object. We manually replace this handler with a
# standard StreamHandler.
# See:
#   https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
_std_formatter = logging.Formatter(_logging_format)
_std_stream_handler = logging.StreamHandler()
_std_stream_handler.setFormatter(_std_formatter)

_std_logger = logging.getLogger()
_std_logger.handlers = [_std_stream_handler]
_std_logger.setLevel(_LOGGING_LEVEL)
std_logger = IndentedLoggerAdapter(_std_logger)

# Turn off the TensorFlow logging.
#   0 = all messages are logged (default behavior)
#   1 = INFO messages are not printed
#   2 = INFO and WARNING messages are not printed
#   3 = INFO, WARNING, and ERROR messages are not printed
# See: https://stackoverflow.com/a/42121886
TensorFlowLoggingLevels = {
    logging.INFO: '0',
    logging.WARNING: '1',
    logging.ERROR: '2',
    'silent': '3',
}
_tf_log_level = TensorFlowLoggingLevels['silent']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = _tf_log_level
tf.logging.set_verbosity(tf.logging.WARN)
