import logging

from python_log_indenter import IndentedLoggerAdapter

import mnist.config as config
import mnist.constants as constants

# General logger.
_LOGGING_LEVEL = config.GeneralConfig.LOGGING_LEVEL
_logging_format = '[%(name)s][%(levelname)s] %(message)s'

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

# Standard logger.
_std_logger = logging.getLogger(constants.LoggerNames.STD)
_std_logger.handlers = [_std_stream_handler]
_std_logger.setLevel(_LOGGING_LEVEL)
std_logger = IndentedLoggerAdapter(_std_logger)

# Data setup logger.
_data_setup_logger = logging.getLogger(constants.LoggerNames.DATA_SETUP)
_data_setup_logger.handlers = [_std_stream_handler]
_data_setup_logger.setLevel(_LOGGING_LEVEL)
data_setup_logger = IndentedLoggerAdapter(_data_setup_logger)
