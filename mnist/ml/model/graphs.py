import tensorflow as tf

import mnist.config as config
import mnist.data.tensorflow_dataset as tf_ds
import mnist.ml.model.cnn_architectures.easynet as easynet
import mnist.ml.model.graph_components as gc
import mnist.ml.model.naming as naming


def build_training_graph(training_dataset_def):
    graph = tf.Graph()
    with graph.as_default():
        training_set = \
            tf_ds.training_set_from_dataset_definition(training_dataset_def)

        input_images, input_labels = gc.input_pipeline(training_set)

        with tf.name_scope(naming.Names.CONVOLUTIONAL_BACKBONE_SCOPE):
            image_feature_map = easynet.easynet(input_images=input_images,
                                                training_flag=True)

        with tf.name_scope(naming.Names.FEATURE_PROCESSING_SCOPE):
            feature_vector = gc.feature_processing(image_feature_map,
                                                   training_flag=True)
            logits = gc.logits_from_feature_vector(feature_vector)

        with tf.name_scope(naming.Names.LOSS_SCOPE):
            loss = tf.losses.sparse_softmax_cross_entropy(labels=input_labels,
                                                          logits=logits)

        # tf.summary.scalar(Names.TRAINING_LOSS_SUMMARY, loss,
        #                   collections=[Names.TRAINING_SUMMARY_COLLECTION])

        optimizer = tf.train.GradientDescentOptimizer(
            config.TrainingConfig.LEARNING_RATE)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            _train_op = optimizer.minimize(loss=loss,
                                           name=naming.Names.TRAINING_OPERATION)

    return graph


# def build_evaluation_graph():
#     data_format = GeneralConfig.DATA_FORMAT
#
#     graph = tf.Graph()
#     with graph.as_default():
#         input_images, input_labels = input_pipeline(data_format)
#
#         with tf.name_scope(Names.CONVOLUTIONAL_BACKBONE_SCOPE):
#             image_feature_map = easynet(input_images=input_images,
#                                         training_flag=False,
#                                         data_format=data_format)
#         with tf.name_scope(Names.FEATURE_PROCESSING_SCOPE):
#             feature_vector = common_feature_processing(image_feature_map, training_flag=False)
#             with tf.variable_scope(Names.CALIBRATION_ONLY_SCOPE):
#                 calibration_only_feature_vector = calibration_only_layers(feature_vector,
#                                                                           training_flag=False)
#             logits = logits_from_feature_vector(calibration_only_feature_vector)
#
#         with tf.name_scope(Names.LOSS_SCOPE):
#             loss = tf.losses.sparse_softmax_cross_entropy(labels=input_labels, logits=logits)
#         tf.identity(loss, Names.EVALUATION_LOSS)
#
#         append_evaluation_outputs(logits, input_labels)
#
#     return graph


# def build_inference_graph():
#     data_format = GeneralConfig.DATA_FORMAT
#
#     graph = tf.Graph()
#     with graph.as_default():
#         image_shape = get_image_shape(data_format)
#         images_tensor_shape = (None,) + image_shape
#         images_placeholder = tf.placeholder(dtype=Constants.IMAGE_DATA_TYPE,
#                                             shape=images_tensor_shape,
#                                             name=Names.IMAGES_PLACEHOLDER)
#
#         with tf.name_scope(Names.CONVOLUTIONAL_BACKBONE_SCOPE):
#             image_feature_map = easynet(input_images=images_placeholder,
#                                         training_flag=False,
#                                         data_format=data_format)
#         with tf.name_scope(Names.FEATURE_PROCESSING_SCOPE):
#             feature_vector = common_feature_processing(image_feature_map, training_flag=False)
#             with tf.variable_scope(Names.CALIBRATION_ONLY_SCOPE):
#                 calibration_only_feature_vector = calibration_only_layers(feature_vector,
#                                                                           training_flag=False)
#             logits = logits_from_feature_vector(calibration_only_feature_vector)
#
#         with tf.name_scope(Names.OUTPUT_SCOPE):
#             # Probabilities over the classes.
#             probabilities = tf.nn.softmax(logits, name=Names.PROBABILITIES)
#
#             # The chosen class.
#             prediction = tf.argmax(input=logits, axis=1, name=Names.PREDICTION)
#
#         tf.add_to_collection(Names.OUTPUT_COLLECTION, probabilities)
#         tf.add_to_collection(Names.OUTPUT_COLLECTION, prediction)
#
#     return graph


# def build_plotting_graph():
#     """Utility graph only used to plot."""
#     graph = tf.Graph()
#     with graph.as_default():
#         avg_batch_loss_placeholder = tf.placeholder(tf.float32,
#                                                     shape=(),
#                                                     name=Names.AVG_BATCH_LOSS_PLACEHOLDER)
#         accuracy_placeholder = tf.placeholder(tf.float32,
#                                               shape=(),
#                                               name=Names.ACCURACY_PLACEHOLDER)
#         calibration_losses_placeholder = tf.placeholder(tf.float32,
#                                                         name=Names.CALIBRATION_LOSSES_PLACEHOLDER)
#
#         tf.summary.scalar(Names.AVG_BATCH_LOSS_SUMMARY,
#                           avg_batch_loss_placeholder,
#                           collections=[Names.EVALUATION_SUMMARY_COLLECTION])
#         tf.summary.scalar(Names.ACCURACY_SUMMARY,
#                           accuracy_placeholder,
#                           collections=[Names.EVALUATION_SUMMARY_COLLECTION])
#         tf.summary.histogram(Names.CALIBRATION_LOSS_SUMMARY,
#                              calibration_losses_placeholder,
#                              collections=[Names.CALIBRATION_SUMMARY_COLLECTION])
#     return graph
