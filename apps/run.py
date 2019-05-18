import json
import os

import tensorflow as tf

import mnist.config as config
import mnist.constants as constants

import mnist.ml.model.graphs as graphs
import mnist.ml.model.naming as naming


def main():
    training_set_def_path = os.path.join(
        config.GeneralConfig.DATASET_DEF_DIR,
        constants.DatasetFilenames[constants.Constants.TEST_SET_NAME]
    )
    with open(training_set_def_path, 'r') as f:
        training_set_def = json.load(f)

    training_set_def = list(training_set_def)[:10]

    training_graph = graphs.build_training_graph(training_set_def)

    with tf.Session(graph=training_graph) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        dataset_init_op \
            = sess.graph.get_operation_by_name(naming.Names.DATASET_INIT_OP)
        sess.run(dataset_init_op)

        train_op = \
            sess.graph.get_operation_by_name(naming.Names.TRAINING_OPERATION)

        train_loss = sess.graph.get_collection(tf.GraphKeys.LOSSES)[0]

        for _ in range(5):
            _, loss_out = sess.run([train_op, train_loss])
            print(loss_out)


if __name__ == '__main__':
    main()