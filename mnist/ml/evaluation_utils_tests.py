import json
import os
import unittest

import numpy as np
from pyfakefs import fake_filesystem_unittest

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


class TestSaveEvaluationResults(fake_filesystem_unittest.TestCase):
    _DATA_DIR = '/my/base/dir'

    def setUp(self):
        self.setUpPyfakefs()
        os.makedirs(self._DATA_DIR)

    def test_no_existing_file_success(self):
        eval_results_path = os.path.join(self._DATA_DIR, 'foo.json')
        eval_results = {'bar': 0, 'baz': 1}

        self.assertFalse(os.path.exists(eval_results_path))

        eval_utils.save_evaluation_results(eval_results_path, eval_results)

        self.assertTrue(os.path.exists(eval_results_path))

        with open(eval_results_path, 'r') as f:
            loaded_eval_results = json.load(f)
        self.assertListEqual(loaded_eval_results, [eval_results])

    def test_existing_file_success(self):
        eval_results_path = os.path.join(self._DATA_DIR, 'foo.json')
        eval_results = {'bar': 0, 'baz': 1}
        existing_results = [
            {'a': 0},
            {'b': 1},
        ]

        expected_eval_results = [
            {'a': 0},
            {'b': 1},
            {'bar': 0, 'baz': 1}
        ]

        self.fs.create_file(eval_results_path,
                            contents=json.dumps(existing_results))

        self.assertTrue(os.path.exists(eval_results_path))

        eval_utils.save_evaluation_results(eval_results_path, eval_results)

        self.assertTrue(os.path.exists(eval_results_path))

        with open(eval_results_path, 'r') as f:
            loaded_eval_results = json.load(f)
        self.assertListEqual(loaded_eval_results, expected_eval_results)


if __name__ == '__main__':
    unittest.main()
