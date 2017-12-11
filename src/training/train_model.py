import tensorflow as tf
import pyprind
import numpy as np
import argparse
import yaml
from src.common import consts, paths
import os
from os import listdir
from os.path import isfile, join
from src.data_preparation import dataset


def get_data_iter(sess_, tf_records_paths_, buffer_size=20, batch_size=64):
    ds_, file_names_ = dataset.features_dataset()
    ds_iter = ds_.shuffle(buffer_size).repeat().batch(batch_size).make_initializable_iterator()
    sess_.run(ds_iter.initializer, feed_dict={file_names_: tf_records_paths_})
    return ds_iter.get_next()


# utility functions for weight and bias init
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# tensorboard + layer utilities
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # reshape input_tensor if needed
        input_shape = input_tensor.get_shape()
        if len(input_shape) == 4:
            ndims = np.int(np.product(input_shape[1:]))
            input_tensor = tf.reshape(input_tensor, [-1, ndims])
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
        tf.summary.histogram(layer_name + '/activations', activations)

    return activations


def linear_model(model_layer_):
    _x = tf.placeholder(dtype=tf.float32, shape=(None, model_layer_[0]), name="x")
    _y = tf.placeholder(dtype=tf.int32, shape=(None), name="y")

    _y_ = fc_layer(_x, input_dim=2048, output_dim=5270, layer_name='FC_1', act=tf.identity)
    return _x, _y, _y_


def get_tfrecrods_files(input_):
    if input_.endswith(".tfrecord"):
        return [input_]
    else:
        return [join(input_, f) for f in listdir(input_) if isfile(join(input_, f)) and f.endswith(".tfrecord")]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c', dest="config_filename", type=str, required=True,
                        help='the config file name must be provide')
    args = parser.parse_args()

    with open(args.config_filename, 'r') as yml_file:
        cfg = yaml.load(yml_file)

    BATCH_SIZE = cfg["TRAIN"]["BATCH_SIZE"]
    EPOCHS_COUNT = cfg["TRAIN"]["EPOCHS_COUNT"]
    LEARNING_RATE = cfg["TRAIN"]["LEARNING_RATE"]
    TRAIN_TF_RECORDS = str(cfg["TRAIN"]["TRAIN_TF_RECORDS"])

    EVAL_BATCH_SIZE = cfg["TRAIN"]["EVAL_BATCH_SIZE"]
    EVAL_TF_RECORDS = str(cfg["TRAIN"]["EVAL_TF_RECORDS"])

    MODEL_NAME = cfg["MODEL"]["MODEL_NAME"]
    MODEL_LAYERS = cfg["MODEL"]["MODEL_LAYERS"]

    print(TRAIN_TF_RECORDS)
    training_files = get_tfrecrods_files(TRAIN_TF_RECORDS)
    eval_files = get_tfrecrods_files(EVAL_TF_RECORDS)

    print("Training model {}".format(MODEL_NAME))
    print("Training data {}".format(training_files))
    print("Training batch size {}".format(BATCH_SIZE))

    print("Evaluation data {}".format(eval_files))
    print("Evaluation batch size {}".format(EVAL_BATCH_SIZE))


    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
        next_train_batch = get_data_iter(sess, training_files, batch_size=BATCH_SIZE)

        next_eval_batch = get_data_iter(sess, eval_files, batch_size=EVAL_BATCH_SIZE)

        x, y, y_ = linear_model(MODEL_LAYERS)

        ###
        # loss and eval functions
        ###

        with tf.name_scope('cross_entropy'):
            cross_entropy_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_)
            cross_entropy = tf.reduce_mean(cross_entropy_i)
            tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_, 1, output_type=tf.int32), y)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        # training step (NOTE: improved optimiser and lower learning rate; needed for more complex model)
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        # Merge all the summaries and write them out to /summaries/conv (by default)
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR, MODEL_NAME, str(LEARNING_RATE), 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR, MODEL_NAME, str(LEARNING_RATE), 'test'))

        # sess.run(tf.global_variables_initializer()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()

        bar = pyprind.ProgBar(EPOCHS_COUNT, update_interval=1, width=60)
        # main training loop
        for epoch in range(EPOCHS_COUNT):
            batch_examples = sess.run(next_train_batch)
            batch_inception_features = batch_examples[consts.INCEPTION_OUTPUT_FIELD]
            batch_y = batch_examples[consts.LABEL_ONE_HOT_FIELD]

            _, summary = sess.run([optimizer, merged], feed_dict={
                                      x: batch_inception_features,
                                      y: batch_y
                                  })

            train_writer.add_summary(summary, epoch)

            # Record summaries and test-set accuracy
            if epoch % 10 == 0 or epoch == EPOCHS_COUNT:
                eval_batch_examples = sess.run(next_eval_batch)
                eval_batch_features = eval_batch_examples[consts.INCEPTION_OUTPUT_FIELD]
                eval_batch_y = eval_batch_examples[consts.LABEL_ONE_HOT_FIELD]

                dev_summaries = sess.run(merged, feed_dict={
                                          x: eval_batch_features,
                                          y: eval_batch_y
                                      })
                test_writer.add_summary(dev_summaries, epoch)

                saver.save(sess, os.path.join(paths.CHECKPOINTS_DIR, MODEL_NAME, str(LEARNING_RATE), MODEL_NAME),
                           latest_filename=MODEL_NAME + '_latest')
            bar.update()
