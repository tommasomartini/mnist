import tensorflow as tf

import mnist.constants as constants


def load_image(image_path):
    raw_image = tf.io.read_file(image_path)
    return raw_image


def preprocess_image(raw_image):
    tensor_image = tf.image.decode_png(
        raw_image,
        channels=constants.Constants.MNIST_IMAGE_CHANNELS)
    float_image = tf.cast(tensor_image,
                          dtype=constants.Constants.IMAGE_DATA_TYPE)
    normalized_image = float_image / 255.0
    return normalized_image


def load_and_preprocess_image(image_path):
    raw_image = load_image(image_path)
    preprocessed_image = preprocess_image(raw_image)
    return preprocessed_image
