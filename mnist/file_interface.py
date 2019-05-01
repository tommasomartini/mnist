import hashlib
import os
from enum import Enum

from mnist.config import GeneralConfig
from mnist.constants import Constants


class _SampleFileType(Enum):
    """Describes the files associated with each dataset sample."""
    IMAGE = 0
    METADATA = 1


################################################################################
# Generic ######################################################################
################################################################################

def _get_hash_subdir(input_str, num_levels):
    """Returns the hash-subfolder for the provided string.

    The input string is hashed using the SHA-1 algorithm
    (see: https://en.wikipedia.org/wiki/SHA-1). The relative path is created by
    concatenating the last bytes in hex format.

    Example:
        input_str = 'foo'
        num_levels = 2
    The SHA-1 hash string is '0beec7b5ea3f0fdbc95d0dd47f3c5bc275da8a33', that
    is 40 characters and 20 bytes. The last two bytes are '8a' and '33', hence
    the hash-subfolder will be '8a/33'.

    Args:
        input_str (str): Input string to map to a hash-subfolder.
        num_levels (int): Number of subfolder levels generated.

    Return:
        The hash-subfolder of the provided input as a relative path.

    Raises:
        ValueError: Too many folder levels requested.
        ValueError: Number of folder levels must be strictly positive.
    """
    if num_levels < 1:
        raise ValueError('Number of subfolder levels must be strictly positive,'
                         'provided was: {}'.format(num_levels))

    hash_obj = hashlib.sha1(input_str.encode())
    if num_levels > hash_obj.digest_size:
        raise ValueError('Requested {} subfolder levels, '
                         'but current hashing function only allows '
                         '{}'.format(num_levels, hash_obj.digest_size))

    hash_str = hash_obj.hexdigest()

    chars_per_byte = 2
    rev_subfolders = [
        hash_str[idx: idx - chars_per_byte: -1]
        for idx
        in range(-1, - num_levels * chars_per_byte, - chars_per_byte)]
    rev_hash_subdir = os.path.join(*rev_subfolders)
    hash_subdir = rev_hash_subdir[::-1]
    return hash_subdir


def mkdir_if_not_exists(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


################################################################################
# File path getters ############################################################
################################################################################

def _get_sample_file_path(data_dir, sample_id, file_type):
    if file_type == _SampleFileType.IMAGE:
        file_ext = Constants.IMAGE_EXTENSION
    elif file_type == _SampleFileType.METADATA:
        file_ext = Constants.METADATA_EXTENSION
    else:
        raise ValueError('Unknown sample file type {}'.format(file_type))

    hash_subdir = _get_hash_subdir(
        sample_id,
        num_levels=GeneralConfig.NUM_HASH_SUBDIR_LEVELS)
    filename = str(sample_id) + os.path.extsep + file_ext
    filepath = os.path.join(data_dir, hash_subdir, filename)
    return filepath


def get_image_path(data_dir, sample_id):
    image_path = _get_sample_file_path(data_dir,
                                       sample_id,
                                       file_type=_SampleFileType.IMAGE)
    return image_path


def get_metadata_path(data_dir, sample_id):
    meta_path = _get_sample_file_path(data_dir,
                                      sample_id,
                                      file_type=_SampleFileType.METADATA)
    return meta_path
