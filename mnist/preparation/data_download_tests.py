import os
import unittest

import numpy as np
from pyfakefs import fake_filesystem_unittest

import mnist.preparation.data_download as dd


class TestSaveSampleImage(fake_filesystem_unittest.TestCase):
    _DATA_DIR = '/my/base/dir'
    _SAMPLE_ID = 'foo'

    def setUp(self):
        self.setUpPyfakefs()
        self._image = np.arange(6).reshape((3, 2)).astype(np.float)

    def test_success(self):
        image_path = '/my/base/dir/8a/33/foo.png'
        dd._save_sample_image(data_dir=self._DATA_DIR,
                              sample_id=self._SAMPLE_ID,
                              raw_image=self._image,
                              dry=False)
        self.assertTrue(os.path.exists(image_path))

    def test_folder_exists_success(self):
        image_path = '/my/base/dir/8a/33/foo.png'
        os.makedirs(os.path.dirname(image_path))
        dd._save_sample_image(data_dir=self._DATA_DIR,
                              sample_id=self._SAMPLE_ID,
                              raw_image=self._image,
                              dry=False)
        self.assertTrue(os.path.exists(image_path))

    def test_dry_success(self):
        image_path = '/my/base/dir/8a/33/foo.png'
        dd._save_sample_image(data_dir=self._DATA_DIR,
                              sample_id=self._SAMPLE_ID,
                              raw_image=self._image,
                              dry=True)
        self.assertFalse(os.path.exists(image_path))


class TestSaveSampleMetadata(fake_filesystem_unittest.TestCase):
    _DATA_DIR = '/my/base/dir'
    _SAMPLE_ID = 'foo'
    _LABEL = 5

    def setUp(self):
        self.setUpPyfakefs()

    def test_success(self):
        meta_path = '/my/base/dir/8a/33/foo.json'
        dd._save_sample_metadata(data_dir=self._DATA_DIR,
                                 sample_id=self._SAMPLE_ID,
                                 label=self._LABEL,
                                 dry=False)
        self.assertTrue(os.path.exists(meta_path))

    def test_folder_exists_success(self):
        meta_path = '/my/base/dir/8a/33/foo.json'
        dd._save_sample_metadata(data_dir=self._DATA_DIR,
                                 sample_id=self._SAMPLE_ID,
                                 label=self._LABEL,
                                 dry=False)
        self.assertTrue(os.path.exists(meta_path))

    def test_dry_success(self):
        meta_path = '/my/base/dir/8a/33/foo.json'
        dd._save_sample_metadata(data_dir=self._DATA_DIR,
                                 sample_id=self._SAMPLE_ID,
                                 label=self._LABEL,
                                 dry=True)
        self.assertFalse(os.path.exists(meta_path))


if __name__ == '__main__':
    unittest.main()
