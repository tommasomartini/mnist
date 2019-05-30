import tensorflow as tf
import tensorflow.keras.activations as kactivations
import tensorflow.keras.layers as klayers

import mnist.config as config
import mnist.constants as constants
import mnist.ml.model.naming as naming


def input_pipeline(dataset):
    dataset_iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                       dataset.output_shapes)
    _dataset_init_op = dataset_iterator.make_initializer(
        dataset,
        name=naming.Names.DATASET_INIT_OP)
    input_images, input_labels = dataset_iterator.get_next()
    return input_images, input_labels


def feature_processing(input_feature_map, training_flag):
    feature_map_flat = klayers.Flatten(name=naming.Names.FEATURE_MAP_FLAT)(
        input_feature_map)
    fc1 = klayers.Dense(units=1024,
                        activation=kactivations.relu,
                        name=naming.Names.FC1)(feature_map_flat)
    dropout_fc1 = klayers.Dropout(rate=0.5,
                                  seed=config.GeneralConfig.RANDOM_SEED,
                                  name=naming.Names.DROPOUT_FC1)(
        fc1,
        training=training_flag)
    fc2 = klayers.Dense(units=1024,
                        activation=kactivations.relu,
                        name=naming.Names.FC2)(dropout_fc1)
    dropout_fc2 = klayers.Dropout(rate=0.5,
                                  seed=config.GeneralConfig.RANDOM_SEED,
                                  name=naming.Names.DROPOUT_FC2)(
        fc2,
        training=training_flag)
    return dropout_fc2


def logits_from_feature_vector(input_feature_vector):
    logits = klayers.Dense(units=constants.Constants.MNIST_NUM_CLASSES,
                           name=naming.Names.LOGITS)(input_feature_vector)
    return logits


def evaluation_outputs(logits, input_labels):
    with tf.name_scope(naming.Names.OUTPUT_SCOPE):
        # Probabilities over the classes.
        probabilities = klayers.Softmax(name=naming.Names.PROBABILITIES)(logits)

        # The chosen class.
        prediction = tf.argmax(input=logits,
                               axis=1,
                               name=naming.Names.PREDICTION)

        # How many predictions match the labels (in this batch).
        is_prediction_correct = tf.equal(tf.cast(prediction, tf.int32),
                                         input_labels)
        num_correct_predictions = \
            tf.reduce_sum(tf.cast(is_prediction_correct, tf.float32),
                          name=naming.Names.NUM_CORRECT_PREDICTIONS)

        # The number of samples in the current batch.
        batch_size = tf.shape(probabilities, name=naming.Names.BATCH_SIZE)[0]

    tf.add_to_collection(naming.Names.OUTPUT_COLLECTION, probabilities)
    tf.add_to_collection(naming.Names.OUTPUT_COLLECTION, prediction)
    tf.add_to_collection(naming.Names.OUTPUT_COLLECTION,
                         num_correct_predictions)
    tf.add_to_collection(naming.Names.OUTPUT_COLLECTION, batch_size)


def inference_outputs(logits):
    with tf.name_scope(naming.Names.OUTPUT_SCOPE):
        # Probabilities over the classes.
        probabilities = \
            klayers.Softmax(name=naming.Names.PROBABILITIES)(logits)

        # The chosen class.
        prediction = tf.argmax(input=logits,
                               axis=1,
                               name=naming.Names.PREDICTION)

    tf.add_to_collection(naming.Names.INFERENCE_OUTPUT_COLLECTION,
                         prediction)
    tf.add_to_collection(naming.Names.INFERENCE_OUTPUT_COLLECTION,
                         probabilities)
