import json

import tensorflow as tf

import mnist.config as config
import mnist.data.preprocessing.preprocessing as preproc
import mnist.file_interface as fi
import mnist.paths as paths

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _images_and_labels_from_dataset_definition(dataset_definition):
    data_dir = paths.BasePaths.DATA_DIR

    all_image_paths = []
    all_sample_labels = []
    for sample_id in dataset_definition:
        # Get the path to the image (TF will take care of loading).
        image_path = fi.get_image_path(data_dir, sample_id)
        all_image_paths.append(image_path)

        # Load the sample label.
        meta_path = fi.get_metadata_path(data_dir, sample_id)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        label = fi.MetadataReader.get_label(meta)
        all_sample_labels.append(label)

    return all_image_paths, all_sample_labels


def training_set_from_dataset_definition(dataset_definition):
    num_samples = len(dataset_definition)

    all_image_paths, all_sample_labels = \
        _images_and_labels_from_dataset_definition(dataset_definition)
    dataset = tf.data.Dataset.from_tensor_slices(
        (all_image_paths, all_sample_labels)) \
        .map(preproc.load_and_preprocess_sample,
             num_parallel_calls=AUTOTUNE) \
        .shuffle(buffer_size=num_samples) \
        .batch(config.ExperimentConfig.BATCH_SIZE_TRAINING,
               drop_remainder=True) \
        .prefetch(buffer_size=AUTOTUNE)

    return dataset


def _evaluation_set_from_dataset_definition(dataset_definition, batch_size):
    all_image_paths, all_sample_labels = \
        _images_and_labels_from_dataset_definition(dataset_definition)
    dataset = tf.data.Dataset.from_tensor_slices(
        (all_image_paths, all_sample_labels)) \
        .map(preproc.load_and_preprocess_sample,
             num_parallel_calls=AUTOTUNE) \
        .batch(batch_size,
               drop_remainder=False) \
        .prefetch(buffer_size=AUTOTUNE)

    return dataset


def validation_set_from_dataset_definition(dataset_definition):
    batch_size = config.ExperimentConfig.BATCH_SIZE_VALIDATION
    dataset = _evaluation_set_from_dataset_definition(dataset_definition,
                                                      batch_size)
    return dataset


def evaluation_set_from_dataset_definition(dataset_definition):
    batch_size = config.ExperimentConfig.BATCH_SIZE_TEST
    dataset = _evaluation_set_from_dataset_definition(dataset_definition,
                                                      batch_size)
    return dataset
