import json
import os

import mnist.constants as constants
import mnist.file_interface as fi
import mnist.paths as paths


def main():
    data_dir = paths.BasePaths.DATA_DIR
    if not os.path.exists(data_dir):
        raise IOError('Base data folder {} does not exist'.format(data_dir))

    dataset_def_dir = paths.BasePaths.DATASET_DEF_DIR
    if not os.path.exists(dataset_def_dir):
        raise IOError('Dataset definition folder {} '
                      'does not exist'.format(dataset_def_dir))

    checked_datasets = {}

    for split_name, filename in constants.DatasetFilenames.items():
        print('Dataset {}'.format(split_name))
        dataset_path = os.path.join(dataset_def_dir, filename)
        if not os.path.exists(dataset_path):
            print('  No dataset {} at path {}'.format(split_name, dataset_path))
            continue

        with open(dataset_path, 'r') as f:
            sample_ids = set(json.load(f))

        print('  {} samples'.format(len(sample_ids)))

        for checked_name, checked_samples in checked_datasets.items():
            intersec = sample_ids & checked_samples
            if len(intersec) > 0:
                raise ValueError('{} samples are both in {} '
                                 'and {}'.format(len(intersec),
                                                 checked_name,
                                                 split_name))
        checked_datasets[split_name] = sample_ids

        class_samples = {}
        for sample_id in sample_ids:
            image_path = fi.get_image_path(data_dir, sample_id)
            if not os.path.exists(image_path):
                raise ValueError('No image for sample {} '
                                 'at {}'.format(sample_id, image_path))

            meta_path = fi.get_metadata_path(data_dir, sample_id)
            if not os.path.exists(meta_path):
                raise ValueError('No metadata file for sample {} '
                                 'at {}'.format(sample_id, meta_path))

            with open(meta_path, 'r') as f:
                meta = json.load(f)

            label = fi.MetadataReader.get_label(meta)
            class_samples.setdefault(label, set()).add(sample_id)

        for label, label_samples in class_samples.items():
            print('    Label {}: {} samples'.format(label, len(label_samples)))


if __name__ == '__main__':
    main()
