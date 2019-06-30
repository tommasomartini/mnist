import json
import os
import random

import numpy as np
from tqdm import tqdm

import mnist.config as config
import mnist.constants as constants
import mnist.file_interface as fi
from mnist.custom_utils.logger import std_logger as logger

random.seed(config.GeneralConfig.RANDOM_SEED)


def _gather_samples_by_class(data_dir, silent=False):
    """Returns a dict of all the sample ids, divided by class label.

    Args:
        data_dir (str): Path to the base data folder to load the samples from.
        silent (bool, optional): If True, no message is print out. Defaults to
            False.

    Returns:
        A dict mapping a class label to all the sample ids belonging to that
        class.
    """
    all_meta_paths = fi.get_all_metadata_filepaths_from_dir(data_dir)

    logger.debug('{} files to fetch from disk'.format(len(all_meta_paths)))

    class_samples = {}
    for meta_path in tqdm(all_meta_paths,
                          desc='Collecting all samples',
                          disable=silent):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        sample_id = fi.MetadataReader.get_id(meta)
        sample_label = fi.MetadataReader.get_label(meta)

        sample_set = class_samples.setdefault(sample_label, set())
        sample_set.add(sample_id)

    return class_samples


def generate_datasets(data_dir,
                      dataset_def_dir,
                      data_split_weights,
                      silent=False,
                      dry=False):
    if set(data_split_weights.keys()) != set(constants.DatasetFilenames.keys()):
        raise ValueError('Names of the split sets do not match the known names'
                         'for filename mapping')

    if not os.path.exists(data_dir):
        raise IOError('Folder {} does not exist'.format(data_dir))

    if not os.path.exists(dataset_def_dir):
        raise IOError('Folder {} does not exist'.format(dataset_def_dir))

    # Early check that the files we are going to generate do not exist.
    if not dry:
        for split_name, filename in constants.DatasetFilenames.items():
            dataset_path = os.path.join(dataset_def_dir, filename)
            if os.path.exists(dataset_path):
                raise ValueError('Dataset {} '
                                 'already exists'.format(dataset_path))

    if dry:
        logger.warning('Dry run!')

    class_samples = _gather_samples_by_class(data_dir, silent)

    # Convert the data split weights into floats summing up to 1.
    sum_weights = sum(list(data_split_weights.values()))
    data_split_fracs = {
        split_name: float(frac) / sum_weights
        for split_name, frac
        in data_split_weights.items()
    }

    logger.debug('Data splits')
    for split_name, split_frac in data_split_fracs.items():
        logger.debug('  {}: {}'.format(split_name, split_frac))

    datasets = {
        split_name: set()
        for split_name in data_split_fracs.keys()
    }
    for label, sample_ids in class_samples.items():
        logger.debug('Class {}: {} samples'.format(label, len(sample_ids)))
        _for_class_label = 'for_class_label'
        logger.push(_for_class_label).add()

        num_samples = len(sample_ids)
        assigned_sample_ids = set()
        for split_idx, \
            (split_name, frac) in enumerate(data_split_fracs.items()):
            logger.debug(split_name)
            logger.push(split_name).add()

            split_size = int(np.round(frac * num_samples))

            logger.debug('{} samples to assign'.format(split_size))

            if split_size == 0:
                raise ValueError('0 samples in {}. Consider adding samples '
                                 'or using a different '
                                 'dataset split'.format(split_name))

            unassigned_sample_ids = sample_ids - assigned_sample_ids

            logger.debug('{} unassigned samples'
                         ''.format(len(unassigned_sample_ids)))

            if split_idx == len(datasets) - 1:
                # Last split: use all the remaining samples.
                logger.debug('Last split {}: assign all '
                             'the remaining samples'.format(split_name))
                datasets[split_name].update(unassigned_sample_ids)
            else:
                split_samples = random.sample(unassigned_sample_ids, split_size)
                assigned_sample_ids.update(split_samples)
                datasets[split_name].update(split_samples)

                logger.debug('Split {}: '
                             'assign {} samples'.format(split_name,
                                                        len(split_samples)))

            logger.pop(split_name)

        logger.pop(_for_class_label)

    # Save the datasets on disk.
    for split_name, sample_ids in datasets.items():
        logger.info('Dataset {}: {} samples'.format(split_name,
                                                    len(sample_ids)))

        filename = constants.DatasetFilenames[split_name]
        dataset_path = os.path.join(dataset_def_dir, filename)
        sorted_sample_ids = sorted(list(sample_ids))
        if not dry:
            with open(dataset_path, 'w') as f:
                json.dump(sorted_sample_ids, f, indent=4)
            logger.info('Dataset {} stored at {}'.format(split_name, dataset_path))
