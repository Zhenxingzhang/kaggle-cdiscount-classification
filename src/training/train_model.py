"""
Model file for vgg like model for tiny imagenet using TFrecords, queues, and tensorboard

Assume tfrecords already created (see tfrecords_and_queues.py)

NOTE: There are some key features missing from the this single file example. Most notable is the lack of evaluation.
When working with tfrecords and queues, tensorflow examples suggest separate graphs running training and inference.
The avoids the complication of swapping out the data loading op of just one graph. To achieve this we must introduce
a saver to periodically store the learned parameters and run the evaluation script over this checkpoints
(see https://github.com/tensorflow/tensorflow/tree/r0.7/tensorflow/models/image/cifar10 for a sample)
"""
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
from src.data_preparation.dataset import read_record_to_queue



###
# models and variables
###

# utility functions for weight and bias init
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# conv and pool operations
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


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


def fc_layer(input_tensor, num_units, layer_name, act=tf.nn.relu, dropout=True, keep_prob=None):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.

    Here we read the shape of the incoming tensor (flatten if needed) and use it
    to set shapes of variables
    """

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # reshape input_tensor if needed
        input_shape = input_tensor.get_shape()
        if len(input_shape) == 4:
            ndims = np.int(np.product(input_shape[1:]))
            input_tensor = tf.reshape(input_tensor, [-1, ndims])
        elif len(input_shape) == 2:
            ndims = input_shape[-1].value
        else:
            raise RuntimeError('Strange input tensor shape: {}'.format(input_shape))
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([ndims, num_units])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([num_units])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
        tf.summary.histogram(layer_name + '/activations', activations)
        if dropout:
            activations_drop = tf.nn.dropout(activations, keep_prob)
            return activations_drop

    return activations


def conv_pool_layer(input_tensor, filter_size, num_filters, layer_name, act=tf.nn.relu, pool=True):
    """Reusable code for making a simple conv_pool layer.

    It does a 2D convolution, bias add, and then uses relu to nonlinearize, (optionally) followed by 2x2 max pooling
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.

    Here we read the shape of the incoming tensor and use it to set shapes of variables
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        patches_in = input_tensor.get_shape()[-1].value

        with tf.name_scope('weights'):
            weights = weight_variable([filter_size, filter_size, patches_in, num_filters])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([num_filters])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = conv2d(input_tensor, weights) + biases
            # tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
        # tf.summary.histogram(layer_name + '/activations', activations)
        if pool:
            pooled_activations = max_pool_2x2(activations)
            # tf.summary.histogram(layer_name + '/pooled_activations', pooled_activations)

            return pooled_activations
        else:
            return activations


# any preprocessing you need can be added here, e.g. mean substraction, crops, flips etc.
def pre_process_ims(image):
    # return tf.reshape(tf.cast(image, tf.float32) / 255., (128, 128, 3))
    return tf.cast(image, tf.float32) / 255.


def train(filename):
    BATCH_SIZE = 32
    NUM_STEPS = 100000
    IMAGE_WIDTH = 180
    IMAGE_HEIGHT = 180

    with tf.name_scope('data'):
        # shapes = {"image": (IMAGE_WIDTH, IMAGE_HEIGHT, 3), "label": 1}
        shapes = np.array([IMAGE_WIDTH, IMAGE_HEIGHT, 3])
        images_batch, labels_batch = read_record_to_queue(filename, shapes,
                                                          preproc_func=pre_process_ims, num_epochs=None,
                                                          batch_size=BATCH_SIZE)
        # Display the training images in the visualizer.
        tf.summary.image('images', images_batch)

    with tf.name_scope('dropout_keep_prob'):
        keep_prob = tf.placeholder(tf.float32)  # use keep_prob as a placeholder so can modify later

    # MODEL
    out_1 = conv_pool_layer(images_batch, filter_size=3, num_filters=16, layer_name='conv_1', pool=False)
    out_2 = conv_pool_layer(out_1, filter_size=3, num_filters=16, layer_name='conv_pool_2')
    out_3 = conv_pool_layer(out_2, filter_size=3, num_filters=16, layer_name='conv_3', pool=False)
    out_4 = conv_pool_layer(out_3, filter_size=3, num_filters=32, layer_name='conv_pool_4')
    out_5 = conv_pool_layer(out_4, filter_size=3, num_filters=32, layer_name='conv_pool_5')
    out_6 = conv_pool_layer(out_5, filter_size=3, num_filters=64, layer_name='conv_pool_6')
    out_7 = fc_layer(out_6, num_units=128, layer_name='FC_1', keep_prob=keep_prob, dropout=True)
    out_8 = fc_layer(out_7, num_units=256, layer_name='FC_2', keep_prob=keep_prob, dropout=True)
    y_pred = fc_layer(out_8, num_units=200, layer_name='softmax', act=tf.identity, dropout=False)

    # for monitoring
    with tf.name_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(labels_batch), logits=y_pred)
        loss_mean = tf.reduce_mean(loss)
        tf.summary.scalar('loss', loss_mean)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.squeeze(labels_batch))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_mean)

    sess = tf.Session()
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('log/9layer/train', sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    t0 = time.time()
    loss_avg = 0
    for i in range(NUM_STEPS):
        # pass it in through the feed_dict
        _, loss_val = sess.run([train_op, loss_mean], feed_dict={keep_prob: 0.8})
        loss_avg += loss_val
        if i % 10 == 0:
            time_per_batch = (time.time() - t0) / 10.0
            time_per_image = time_per_batch / BATCH_SIZE
            loss_avg /= 10.0
            print("Step: {}, Mean Loss: {}, Time per batch: {:.2f}, Time per image: {:.4f}".format(i, loss_avg,
                                                                                                   time_per_batch,
                                                                                                   time_per_image))
            loss_avg = 0
            t0 = time.time()

        if i % 50 == 0:
            t1 = time.time()
            summary = sess.run(summary_op, feed_dict={keep_prob: 0.8})
            train_writer.add_summary(summary, i)
            print("Wrote summary in {:.2f} s".format(time.time() - t1))


def validate_train_data(tfrecords_filename, shapes):

    img_batch, label_batch = read_record_to_queue(tfrecords_filename, shapes)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # Let's read off 3 batches just for example
        for i in range(3):
            imgs, labels = sess.run([img_batch, label_batch])
            print(imgs.shape)
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    IMAGE_HEIGHT = 180
    IMAGE_WIDTH = 180

    tf_records_filename = "../input/train-100k.tfrecords"

    image_shape = np.asarray([IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # validate_train_data(tf_records_filename, image_shape)

    train(tf_records_filename)
    print("Hello world!")

