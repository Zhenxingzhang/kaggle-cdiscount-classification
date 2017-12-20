"""
VGG ConvNet Model using TensorFlow-Slim Model Library
"""

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg
import tensorflow as tf
import time


def inference(images, no_classes, dropout_keep_prob,
              weight_decay, is_training=True):
    '''
    Load VGG16 and update last fc layer with correct no of classes
    '''

    # Get VGG16 pretrained model
    # Specify num_classes and create a new fully connected layer to replace
    # the last one
    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
        logits, endpoints = vgg.vgg_16(images,
                                       num_classes=no_classes,
                                       is_training=is_training,
                                       dropout_keep_prob=dropout_keep_prob)

        feature_layer = endpoints['vgg_16/fc8']

        return logits, feature_layer


def loss(logits, labels, summaries=True):
    '''
    Calculate cross entropy loss
    Use sparse_softmax_cross_entropy_with_logits as labels have shape
    [batch_size] (i.e. not one hot version)
    '''
    if summaries:
        tf.summary.histogram('logits', logits)
        tf.summary.histogram('probabilities', tf.nn.sigmoid(logits))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
        tf.summary.scalar('loss', loss)
    return loss


def train(loss, learning_rate, var_list=None):
    '''
    Train model by optimizing Adam algorithm
    Optional var_list to limit variables to update while minimizing loss
    '''
    optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer.minimize(loss, var_list=var_list)


def accuracy(logits, labels):
    '''
    Calculate accuracy of logits at predicting labels
    Add summary to track accuracy on TensorBoard
    Return boolean tensor for correct predictions
    '''
    prediction = tf.to_int64(tf.argmax(logits, 1))
    correct_prediction = tf.equal(prediction, labels)
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                 name='accuracy_op')
    tf.summary.scalar('accuracy', accuracy_op)
    return correct_prediction


def evaluate(sess, data_init_op, accuracy_op, is_training_pl, is_training):
    '''
    Evaluate accuracy of model
    '''

    # Initialise data
    sess.run(data_init_op)

    num_samples, num_correct = 0, 0
    t0 = time.time()

    while True:
        try:
            acc = sess.run(accuracy_op, feed_dict={is_training_pl: is_training})
            num_correct += acc.sum()
            num_samples += acc.shape[0]
        except tf.errors.OutOfRangeError:
            break
    accuracy = float(num_correct) / num_samples

    if is_training:
        print(("Train Accuracy: {:.4f} [timer: {:.2f}s]")
              .format(accuracy, time.time() - t0))
    else:
        print(("Val Accuracy: {:.4f} [timer: {:.2f}s]")
              .format(accuracy, time.time() - t0))
