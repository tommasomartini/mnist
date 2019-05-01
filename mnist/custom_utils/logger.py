import logging

from mnist.config import GeneralConfig

# General logger.
_LOGGING_LEVEL = GeneralConfig.LOGGING_LEVEL
_logging_format = '[%(levelname)s]:%(message)s'

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

std_logger = logging.getLogger()
std_logger.handlers = [_std_stream_handler]
std_logger.setLevel(_LOGGING_LEVEL)
