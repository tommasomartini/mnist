import os
import unittest

from pyfakefs import fake_filesystem_unittest

import mnist.paths as paths


class TestMakeDirIfNotExists(fake_filesystem_unittest.TestCase):

    def setUp(self):
        self.setUpPyfakefs()
        self._base_dir = '/my/base/dir'

    def test_dir_exists_success(self):
        os.makedirs(self._base_dir)
        self.assertTrue(os.path.isdir(self._base_dir))
        paths.mkdir_if_not_exists(self._base_dir)
        self.assertTrue(os.path.isdir(self._base_dir))

    def test_dir_does_not_exist_success(self):
        self.assertFalse(os.path.isdir(self._base_dir))
        paths.mkdir_if_not_exists(self._base_dir)
        self.assertTrue(os.path.isdir(self._base_dir))


if __name__ == '__main__':
    unittest.main()
