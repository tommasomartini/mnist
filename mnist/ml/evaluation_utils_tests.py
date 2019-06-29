import unittest

import numpy as np

import mnist.ml.evaluation_utils as eval_utils


class TestEvaluationAccumulator(unittest.TestCase):
    def test_success(self):
        batches = [
            (0, 0.1, 1, 4),
            (1, 0.9, 3, 4),
            (2, 0.2, 2, 4),
        ]
        expected_avg_loss = 0.1
        expected_accuracy = 0.5

        def _batch_gen():
            for b in batches:
                yield b

        avg_loss, accuracy = eval_utils.evaluation_accumulator(_batch_gen())

        np.testing.assert_almost_equal(avg_loss, expected_avg_loss)
        np.testing.assert_almost_equal(accuracy, expected_accuracy)


if __name__ == '__main__':
    unittest.main()
