from itertools import chain
from itertools import islice
from itertools import starmap

import numpy as np

import mnist.constants as constants


# def batches_per_epoch(dataset_size, batch_size, drop_last=True):
#     float_batches_over_epoch = float(dataset_size) / batch_size
#     if drop_last:
#         num_batches_per_epoch = int(np.floor(float_batches_over_epoch))
#     else:
#         num_batches_per_epoch = int(np.ceil(float_batches_over_epoch))
#     return num_batches_per_epoch


# def epoch_cursor(epoch_idx, batch_idx, num_batches_per_epoch):
#     epoch_cursor_multiplier = constants.Constants.EPOCH_CURSOR_MULTIPLIER
#     if num_batches_per_epoch >= epoch_cursor_multiplier:
#         raise RuntimeError('The number of batches per epoch must be smaller than the epoch '
#                            'cursor multiplier: {} vs {}'.format(num_batches_per_epoch,
#                                                                 epoch_cursor_multiplier))
#     num_done_batches = (epoch_idx * num_batches_per_epoch) + batch_idx + 1
#     fraction_done_epochs = float(num_done_batches) / num_batches_per_epoch
#     epoch_cursor = int(np.round(fraction_done_epochs * epoch_cursor_multiplier))
#     return epoch_cursor

def end_of_epoch_cursor(epoch_idx):
    end_of_epoch_cursor = \
        int((epoch_idx + 1) * constants.Constants.EPOCH_CURSOR_MULTIPLIER)
    return end_of_epoch_cursor
