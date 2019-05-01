import logging

from mnist.config import GeneralConfig

# General logger.
_LOGGING_LEVEL = GeneralConfig.LOGGING_LEVEL
_logging_format = '[%(levelname)s]:%(message)s'
logging.basicConfig(format=_logging_format,
                    level=_LOGGING_LEVEL)
logger = logging.getLogger()
