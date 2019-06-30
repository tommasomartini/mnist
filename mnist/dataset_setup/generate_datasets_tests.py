import json
import unittest

from pyfakefs import fake_filesystem_unittest

import mnist.constants as constants
import mnist.dataset_setup.generate_datasets as gd
import mnist.file_interface as fi


class TestGatherSamplesByClass(fake_filesystem_unittest.TestCase):

    def setUp(self):
        self.setUpPyfakefs()
        self._data_dir = '/my/base/dir'

        self._class_samples = {
            0: {'foo', 'bar'},
            1: {'baz'},
        }

        for label, sample_ids in self._class_samples.items():
            for sample_id in sample_ids:
                meta = {
                    constants.MetadataFields.ID: sample_id,
                    constants.MetadataFields.LABEL: label,
                }
                meta_path = fi.get_metadata_path(self._data_dir, sample_id)
                self.fs.create_file(meta_path,
                                    contents=json.dumps(meta))

    def test_success(self):
        class_samples = gd._gather_samples_by_class(self._data_dir,
                                                    silent=True)
        self.assertDictEqual(class_samples, self._class_samples)


if __name__ == '__main__':
    unittest.main()
