import glob
import json
import os

from tqdm import tqdm

import mnist.file_interface as fi
from mnist.config import GeneralConfig
from mnist.constants import Constants


def generate_datasets(data_dir):
    # The metadata file pattern should be something like '*.json'.
    meta_file_pattern = '*' + os.path.extsep + Constants.METADATA_EXTENSION

    # Relative path like: '*/*/*'.
    # Add as many '*' as the number of subfolders.
    hash_subdir_pattern = tuple([
        '*' for _
        in range(GeneralConfig.NUM_HASH_SUBDIR_LEVELS)
    ])

    # Put the path pattern together into something like:
    #   /path/to/data/*/*/*.json
    path_pieces = (data_dir,) + hash_subdir_pattern + (meta_file_pattern,)
    meta_path_pattern = os.path.join(*path_pieces)
    all_meta_paths = glob.glob(meta_path_pattern)

    samples_by_label = {}
    for meta_path in tqdm(all_meta_paths,
                          desc='Collecting all samples'):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        sample_id = fi.MetadataReader.get_id(meta)
        sample_label = fi.MetadataReader.get_label(meta)

        sample_list = samples_by_label.setdefault(sample_label, [])
        sample_list.append(sample_id)

    for label, samples in samples_by_label.items():
        print('{}: {} samples'.format(label, len(samples)))
