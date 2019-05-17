import tensorflow as tf

import mnist.custom_utils.readonly as ro


class Names(ro.ReadOnly):
    CONV1 = 'conv1'
    BN1 = 'bn1'
    POOL1 = 'pool1'
    CONV2 = 'conv2'
    BN2 = 'bn2'
    POOL2 = 'pool2'


def easynet(input_images, training_flag):
    conv1 = tf.layers.conv2d(inputs=input_images,
                             filters=32,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu,
                             name=Names.CONV1)
    bn1 = tf.layers.batch_normalization(conv1,
                                        axis=1,
                                        training=training_flag,
                                        name=Names.BN1)
    pool1 = tf.layers.max_pooling2d(inputs=bn1,
                                    pool_size=[2, 2],
                                    strides=2,
                                    name=Names.POOL1)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding='same',
                             activation=tf.nn.relu,
                             name=Names.CONV2)
    bn2 = tf.layers.batch_normalization(conv2,
                                        axis=1,
                                        training=training_flag,
                                        name=Names.BN2)
    pool2 = tf.layers.max_pooling2d(inputs=bn2,
                                    pool_size=[2, 2],
                                    strides=2,
                                    name=Names.POOL2)
    feature_map = pool2
    return feature_map
