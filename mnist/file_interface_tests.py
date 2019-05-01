import os
import unittest

from pyfakefs import fake_filesystem_unittest

import mnist.file_interface as fi


################################################################################
# Generic ######################################################################
################################################################################

class TestHashSubdir(unittest.TestCase):
    """These test cases assume the SHA-1 algorithm is used."""

    def test_2_levels_success(self):
        input_str = 'foo'
        num_levels = 2
        expected_hash_subdir = '8a/33'
        hash_subdir = fi._get_hash_subdir(input_str, num_levels)
        self.assertEqual(hash_subdir, expected_hash_subdir)

    def test_3_levels_success(self):
        input_str = 'foo'
        num_levels = 3
        expected_hash_subdir = 'da/8a/33'
        hash_subdir = fi._get_hash_subdir(input_str, num_levels)
        self.assertEqual(hash_subdir, expected_hash_subdir)

    def test_1_level_success(self):
        input_str = 'foo'
        num_levels = 1
        expected_hash_subdir = '33'
        hash_subdir = fi._get_hash_subdir(input_str, num_levels)
        self.assertEqual(hash_subdir, expected_hash_subdir)

    def test_too_many_subfolder_levels(self):
        input_str = 'foo'
        num_levels = 25
        with self.assertRaises(ValueError):
            fi._get_hash_subdir(input_str, num_levels)

    def test_negative_subfolder_levels(self):
        input_str = 'foo'
        num_levels = -2
        with self.assertRaises(ValueError):
            fi._get_hash_subdir(input_str, num_levels)

    def test_zero_subfolder_levels(self):
        input_str = 'foo'
        num_levels = 0
        with self.assertRaises(ValueError):
            fi._get_hash_subdir(input_str, num_levels)


class TestMakeDirIfNotExists(fake_filesystem_unittest.TestCase):

    def setUp(self):
        self.setUpPyfakefs()
        self._base_dir = '/my/base/dir'

    def test_dir_exists_success(self):
        os.makedirs(self._base_dir)
        self.assertTrue(os.path.isdir(self._base_dir))
        fi.mkdir_if_not_exists(self._base_dir)
        self.assertTrue(os.path.isdir(self._base_dir))

    def test_dir_does_not_exist_success(self):
        self.assertFalse(os.path.isdir(self._base_dir))
        fi.mkdir_if_not_exists(self._base_dir)
        self.assertTrue(os.path.isdir(self._base_dir))


################################################################################
# File path getters ############################################################
################################################################################

class TestGetSampleFilePath(unittest.TestCase):
    _BASE_DIR = '/my/base/dir'
    _SAMPLE_ID = 'foo'

    def test_success(self):
        expected_filepath = '/my/base/dir/8a/33/foo.png'
        filepath = fi._get_sample_file_path(data_dir=self._BASE_DIR,
                                            sample_id=self._SAMPLE_ID,
                                            file_type=fi._SampleFileType.IMAGE)
        self.assertEqual(filepath, expected_filepath)

    def test_invalid_sample_file_type(self):
        with self.assertRaises(ValueError):
            fi._get_sample_file_path(data_dir=self._BASE_DIR,
                                     sample_id=self._SAMPLE_ID,
                                     file_type='bar')


class TestGetImagePath(unittest.TestCase):
    _BASE_DIR = '/my/base/dir'
    _SAMPLE_ID = 'foo'

    def test_success(self):
        expected_image_path = '/my/base/dir/8a/33/foo.png'
        image_path = fi.get_image_path(data_dir=self._BASE_DIR,
                                       sample_id=self._SAMPLE_ID)
        self.assertEqual(image_path, expected_image_path)


class TestGetMetadatPath(unittest.TestCase):
    _BASE_DIR = '/my/base/dir'
    _SAMPLE_ID = 'foo'

    def test_success(self):
        expected_meta_path = '/my/base/dir/8a/33/foo.json'
        meta_path = fi.get_metadata_path(data_dir=self._BASE_DIR,
                                         sample_id=self._SAMPLE_ID)
        self.assertEqual(meta_path, expected_meta_path)


################################################################################
# Metadata readers #############################################################
################################################################################

class TestMetadataReader(unittest.TestCase):
    _META = {
        'id': 'foo',
        'label': 5,
    }

    # get_id

    def test_get_id_success(self):
        expected_id = self._META['id']
        id = fi.MetadataReader.get_id(self._META)
        self.assertEqual(id, expected_id)

    def test_get_id_missing_field(self):
        meta = {
            'label': 5,
        }
        with self.assertRaises(KeyError):
            fi.MetadataReader.get_id(meta)

    # get_label

    def test_get_label_success(self):
        expected_label = self._META['label']
        label = fi.MetadataReader.get_label(self._META)
        self.assertEqual(label, expected_label)

    def test_get_label_missing_field(self):
        meta = {
            'id': 'foo',
        }
        with self.assertRaises(KeyError):
            fi.MetadataReader.get_label(meta)
