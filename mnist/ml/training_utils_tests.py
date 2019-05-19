import unittest

import mock

import mnist.ml.training_utils as training_utils


class TestBatchesPerEpoch(unittest.TestCase):

    def test_drop_last_success(self):
        dataset_size = 100
        batch_size = 9
        drop_last = True
        expected_num_batches = 11

        num_batches = training_utils.batches_per_epoch(
            dataset_size=dataset_size,
            batch_size=batch_size,
            drop_last=drop_last)

        self.assertEqual(num_batches, expected_num_batches)

    def test_keep_last_success(self):
        dataset_size = 100
        batch_size = 9
        drop_last = False
        expected_num_batches = 12

        num_batches = training_utils.batches_per_epoch(
            dataset_size=dataset_size,
            batch_size=batch_size,
            drop_last=drop_last)

        self.assertEqual(num_batches, expected_num_batches)

    def test_batch_size_larger_than_dataset(self):
        dataset_size = 10
        batch_size = 100
        drop_last = True

        with self.assertRaises(ValueError):
            training_utils.batches_per_epoch(
                dataset_size=dataset_size,
                batch_size=batch_size,
                drop_last=drop_last)


class TestEpochCursor(unittest.TestCase):

    @mock.patch('mnist.ml.training_utils._CNST')
    def test_first_batch_success(self, cursor_multiplier_mock):
        cursor_multiplier_mock.EPOCH_CURSOR_MULTIPLIER = 1000
        epoch_idx = 0
        batch_idx = 0
        batches_per_epoch = 20
        expected_cursor = 50

        cursor = training_utils.epoch_cursor(
            epoch_idx=epoch_idx,
            batch_idx=batch_idx,
            batches_per_epoch=batches_per_epoch)

        self.assertEqual(cursor, expected_cursor)

    @mock.patch('mnist.ml.training_utils._CNST')
    def test_last_batch_success(self, cursor_multiplier_mock):
        cursor_multiplier_mock.EPOCH_CURSOR_MULTIPLIER = 1000
        epoch_idx = 0
        batches_per_epoch = 20
        batch_idx = batches_per_epoch - 1
        expected_cursor = 1000

        cursor = training_utils.epoch_cursor(
            epoch_idx=epoch_idx,
            batch_idx=batch_idx,
            batches_per_epoch=batches_per_epoch)

        self.assertEqual(cursor, expected_cursor)

    @mock.patch('mnist.ml.training_utils._CNST')
    def test_middle_batch_success(self, cursor_multiplier_mock):
        cursor_multiplier_mock.EPOCH_CURSOR_MULTIPLIER = 1000
        epoch_idx = 3
        batch_idx = 13
        batches_per_epoch = 20
        expected_cursor = 3700

        cursor = training_utils.epoch_cursor(
            epoch_idx=epoch_idx,
            batch_idx=batch_idx,
            batches_per_epoch=batches_per_epoch)

        self.assertEqual(cursor, expected_cursor)

    def test_negative_batch_index(self):
        epoch_idx = 3
        batch_idx = -13
        batches_per_epoch = 20
        with self.assertRaises(ValueError):
            training_utils.epoch_cursor(epoch_idx=epoch_idx,
                                        batch_idx=batch_idx,
                                        batches_per_epoch=batches_per_epoch)

    def test_negative_epoch_index(self):
        epoch_idx = -3
        batch_idx = 13
        batches_per_epoch = 20
        with self.assertRaises(ValueError):
            training_utils.epoch_cursor(epoch_idx=epoch_idx,
                                        batch_idx=batch_idx,
                                        batches_per_epoch=batches_per_epoch)

    def test_invalid_number_of_batches_per_epoch(self):
        epoch_idx = 3
        batch_idx = 13
        batches_per_epoch = -20
        with self.assertRaises(ValueError):
            training_utils.epoch_cursor(epoch_idx=epoch_idx,
                                        batch_idx=batch_idx,
                                        batches_per_epoch=batches_per_epoch)

    @mock.patch('mnist.ml.training_utils._CNST')
    def test_batches_per_epoch_larger_than_multiplier(self,
                                                      cursor_multiplier_mock):
        cursor_multiplier_mock.EPOCH_CURSOR_MULTIPLIER = 1000
        epoch_idx = 3
        batch_idx = 13
        batches_per_epoch = 2000
        with self.assertRaises(ValueError):
            training_utils.epoch_cursor(epoch_idx=epoch_idx,
                                        batch_idx=batch_idx,
                                        batches_per_epoch=batches_per_epoch)

    def test_batch_index_exceeds_batches_per_epoch(self):
        epoch_idx = 3
        batch_idx = 130
        batches_per_epoch = 20
        with self.assertRaises(ValueError):
            training_utils.epoch_cursor(epoch_idx=epoch_idx,
                                        batch_idx=batch_idx,
                                        batches_per_epoch=batches_per_epoch)


class TestEndOfEpochCursor(unittest.TestCase):

    @mock.patch('mnist.ml.training_utils._CNST')
    def test_success(self, cursor_multiplier_mock):
        cursor_multiplier_mock.EPOCH_CURSOR_MULTIPLIER = 1000
        epoch_idx = 3   # end of the 4th epoch
        expected_cursor = 4000
        cursor = training_utils.end_of_epoch_cursor(epoch_idx=epoch_idx)
        self.assertEqual(cursor, expected_cursor)

    @mock.patch('mnist.ml.training_utils._CNST')
    def test_before_first_epoch_success(self, cursor_multiplier_mock):
        cursor_multiplier_mock.EPOCH_CURSOR_MULTIPLIER = 1000
        epoch_idx = -1
        expected_cursor = 0
        cursor = training_utils.end_of_epoch_cursor(epoch_idx=epoch_idx)
        self.assertEqual(cursor, expected_cursor)

    def test_invalid_epoch_index(self):
        epoch_idx = -2
        with self.assertRaises(ValueError):
            training_utils.end_of_epoch_cursor(epoch_idx=epoch_idx)
