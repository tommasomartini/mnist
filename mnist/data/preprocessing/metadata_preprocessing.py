import tensorflow as tf

import mnist.constants as constants


def preprocess_label(label):
    label = tf.cast(label, dtype=constants.Constants.LABEL_DATA_TYPE)
    return label
