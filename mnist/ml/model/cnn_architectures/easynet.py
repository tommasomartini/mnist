import tensorflow as tf
import tensorflow.keras.layers as klayers

import mnist.custom_utils.readonly as ro


class Names(ro.ReadOnly):
    CONV1 = 'conv1'
    BN1 = 'bn1'
    POOL1 = 'pool1'
    CONV2 = 'conv2'
    BN2 = 'bn2'
    POOL2 = 'pool2'


def easynet(input_images, training_flag):
    conv1 = klayers.Conv2D(filters=32,
                           kernel_size=[5, 5],
                           padding='same',
                           activation=tf.nn.relu,
                           name=Names.CONV1)(input_images)
    bn1 = klayers.BatchNormalization(axis=-1,
                                     name=Names.BN1)(conv1,
                                                     training=training_flag)
    pool1 = klayers.MaxPool2D(pool_size=[2, 2],
                              strides=2,
                              name=Names.POOL1)(bn1)
    conv2 = klayers.Conv2D(filters=32,
                           kernel_size=[5, 5],
                           padding='same',
                           activation=tf.nn.relu,
                           name=Names.CONV2)(pool1)
    bn2 = klayers.BatchNormalization(axis=-1,
                                     name=Names.BN2)(conv2,
                                                     training=training_flag)
    pool2 = klayers.MaxPool2D(pool_size=[2, 2],
                              strides=2,
                              name=Names.POOL2)(bn2)
    feature_map = pool2
    return feature_map
