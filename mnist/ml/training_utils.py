import numpy as np

import mnist.constants as constants
from mnist.custom_utils.logger import std_logger as logger


_CNST = constants.Constants


def batches_per_epoch(dataset_size, batch_size, drop_last):
    """Computes the number of entire batches in one epoch.

    Args:
        dataset_size (int): Number of samples in the dataset.
        batch_size (int): Size of each batch.
        drop_last (bool): If True, the final reminder samples are dropped if
            they are not enough to make an entire batch. Otherwise they are
            considered to be part of the last batch.

    Returns:
        The number of entire batches in the epoch.

    Raises:
        ValueError: Batch size larger than the dataset size.
    """
    if batch_size > dataset_size:
        raise ValueError('Batch size larger than the dataset size '
                         '({} > {})'.format(batch_size, dataset_size))

    if dataset_size == batch_size:
        logger.warning('Dataset and batch have the same size: '
                       '{}'.format(batch_size))

    float_batches_over_epoch = float(dataset_size) / batch_size
    if drop_last:
        num_batches_per_epoch = int(np.floor(float_batches_over_epoch))
    else:
        num_batches_per_epoch = int(np.ceil(float_batches_over_epoch))
    return num_batches_per_epoch


def epoch_cursor(epoch_idx, batch_idx, batches_per_epoch):
    """Returns an epoch cursor given the batch index in the current epoch.

    A cursor is a progressive integer number uniquely denoting the latest
    trained batch, such that the cursor associated with the last batch of each
    epoch is a multiple of a given "cursor multiplier".

    Examples:
        multiplier = 1000
        batches_per_epoch = 20

        epoch_idx = 0   # first epoch
        batch_idx = 0   # first batch
        cursor is 1000 / 20 = 50: trained the first batch of the first epoch

        epoch_idx = 0   # first epoch
        batch_idx = 19   # last batch
        cursor is 1000: trained the last batch of the first epoch

        epoch_idx = 3   # forth epoch
        batch_idx = 13   # 14th batch
        cursor is 3700: trained the 14th batch of the 4th epoch
            3000 units from the first 3 epochs
            +
            14 * (1000 / 20) units from the current epoch, including
                             the just-trained batch

    Args:
        epoch_idx (int): 0-based index of the current epoch.
        batch_idx (int): 0-based index of the current batch within the
            current epoch.
        batches_per_epoch (int): Number of batches in the current epoch.

    Returns:
        A integer number representing a cursor.

    Raises:
        ValueError: Index cannot be negative.
        ValueError: Invalid number of batches per epoch.
        ValueError: Number of batches per epoch larger than cursor multiplier.
        ValueError: Batch index exceeds number of batches per epoch.
    """
    multip = _CNST.EPOCH_CURSOR_MULTIPLIER

    if batch_idx < 0:
        raise ValueError('Negative batch index: {}'.format(batch_idx))

    if epoch_idx < 0:
        raise ValueError('Negative epoch index: {}'.format(epoch_idx))

    if batches_per_epoch < 0:
        raise ValueError('Number of batches per epoch cannot be negative. '
                         'Provided was: {}'.format(batches_per_epoch))

    if batches_per_epoch > multip:
        raise ValueError('Number of batches per epoch larger than '
                         'cursor multiplier ({} > {})'.format(batches_per_epoch,
                                                              multip))

    if batch_idx >= batches_per_epoch:
        raise ValueError('Batch index {} exceeds number of batches '
                         'per epoch {}.'.format(batch_idx, batches_per_epoch))

    units_past_epochs = epoch_idx * multip
    units_per_batch = multip / batches_per_epoch
    units_current_epoch = (batch_idx + 1) * units_per_batch
    float_cursor = units_past_epochs + units_current_epoch
    cursor = int(np.round(float_cursor))
    return cursor


def end_of_epoch_cursor(epoch_idx):
    """Returns the cursor relative to the latest trained epoch.

    Args:
        epoch_idx (int): 0-based index of the latest trained epoch.

    Returns:
        The cursor relative to the latest trained epoch.

    Raises:
        ValueError: Negative epoch index.
    """
    if epoch_idx < 0:
        raise ValueError('Epoch index cannot be negative. '
                         'Provided was: {}'.format(epoch_idx))

    cursor = int((epoch_idx + 1) * _CNST.EPOCH_CURSOR_MULTIPLIER)
    return cursor
