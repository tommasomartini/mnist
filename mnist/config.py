"""
This module contains all the settings and configurations.
"""
import mnist.custom_utils.readonly as ro


class GeneralConfig(ro.ReadOnly):
    """Container for general configurations."""
    RANDOM_SEED = 0
    DATA_DIR = '/home/tom/Data/mnist'
