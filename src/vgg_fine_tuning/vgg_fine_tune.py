"""
Fine tune a VGG model pre-trained on ImageNet, using images from the COCO
dataset (consisting of 100 training images and 25 validation images for each
animal type: bear, bird, cat, dog, giraffe, horse, sheep, and zebra)

Fine-tuning:
- Replace and retrain the classifier on top of the ConvNet
- Fine-tune the weights of the pre-trained network via backpropagation
"""

import tensorflow as tf
import learning_utils
import data_utils
import vgg_conv
import argparse
import time
import sys
import os

# Silence compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None
NO_CLASSES = 8
NO_THREADS = 4
BATCH_SIZE = 400
DROPOUT_KEEP_PROB = 0.5
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 0.001
MOVING_AVERAGE_DECAY = 0.9999
LEARNING_RATE_DECAY_FACTOR = 0.8


def main(_):

    # Read COCO data files and labels
    train_files, train_labs = data_utils.read_coco_files_and_labels(
        os.path.join(FLAGS.data_dir, 'train'))
    val_files, val_labs = data_utils.read_coco_files_and_labels(
        os.path.join(FLAGS.data_dir, 'val'))

    graph = tf.Graph()
    with graph.as_default():

        # Preprocess and batch train data
        batched_train_data = data_utils.preprocess_and_batch_data(
            train_files, train_labs, NO_THREADS, BATCH_SIZE)
        # Preprocess and batch val data
        batched_val_data = data_utils.preprocess_and_batch_data(
            val_files, val_labs, NO_THREADS, BATCH_SIZE)

        # Define an iterator that can operator on either dataset
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images/labels
        # as they are automatically pulled out from the iterator queues

        # A reinitializable iterator is defined by its structure
        # We could use the `output_types` and `output_shapes` properties of
        # either `train_data` or `val_data` here because they are compatible
        iterator = tf.contrib.data.Iterator.from_structure(
            batched_train_data.output_types,
            batched_train_data.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_data)
        val_init_op = iterator.make_initializer(batched_val_data)

        # Bool to indicate whether we are in training or test mode
        is_training = tf.placeholder(tf.bool)

        # Add inference op to Graph
        logits, _ = vgg_conv.inference(images, NO_CLASSES,
                                       DROPOUT_KEEP_PROB,
                                       WEIGHT_DECAY,
                                       is_training=True)

        # Add loss op to Graph
        loss_op = vgg_conv.loss(logits, labels)

        # Add op to train last fully connected layer fc8 to Graph
        fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
        fc8_train_op = vgg_conv.train(loss_op, LEARNING_RATE,
                                      var_list=fc8_variables)

        # Add op to train full model to Graph
        full_train_op = vgg_conv.train(loss_op, LEARNING_RATE)

        # Add accuracy op to Graph
        accuracy_op = vgg_conv.accuracy(logits, labels)

    with tf.Session(graph=graph) as sess:
        # Build summary Tensor based on collection of Summaries
        summary_op = tf.summary.merge_all()

        # Instantiate summary writer for training
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir + '/summary',
                                               sess.graph)

        # Restore pre-trained variables from file
        learning_utils.restore_variables_from_checkpoint(
            sess, FLAGS.checkpoint_dir,
            exclude=['fc8', 'Adam'], load_scope='vgg_16')

        # Initialise remaining variables
        learning_utils.initialize_remaining_variables(sess)

        # Finalise Graph by making it read only so no other ops can be added
        tf.get_default_graph().finalize()

        # Train last fc layer for a few epochs
        print("Training Last Fully Connected Layer:")
        for epoch in range(FLAGS.fc8_steps):
            print("Epoch {}/{}".format(epoch + 1, FLAGS.fc8_steps))
            # Initialise iterator with the training set
            # This will go through an entire epoch until iterator is empty
            sess.run(train_init_op)

            count = 0
            while True:
                try:
                    # Train and write summaries
                    t0 = time.time()
                    summary, _, los = sess.run(
                        [summary_op, fc8_train_op, loss_op],
                        feed_dict={is_training: True})

                    summary_writer.add_summary(summary, count)
                    print(("Step: {}, Loss: {}, [timer: {:.2f}s]")
                          .format(count, los, time.time() - t0))
                    count += 1
                except tf.errors.OutOfRangeError:
                    break

            # Evaluate accuracy on train and val data every epoch
            # vgg_conv.evaluate(sess, train_init_op, accuracy_op, is_train=True)
            # vgg_conv.evaluate(sess, val_init_op, accuracy_op, is_train=False)

        # Train full model continuing with the same weights
        print("Training Full Model:")
        for epoch in range(FLAGS.steps):
            print("Epoch {}/{}".format(epoch + 1, FLAGS.steps))
            sess.run(train_init_op)

            count = 0
            while True:
                try:
                    # Train and write summaries
                    t0 = time.time()
                    summary, _, los = sess.run(
                        [summary_op, full_train_op, loss_op],
                        feed_dict={is_training: True})

                    summary_writer.add_summary(summary, count)
                    print(("Step: {}, Loss: {}, [timer: {:.2f}s]")
                          .format(count, los, time.time() - t0))
                    count += 1
                except tf.errors.OutOfRangeError:
                    break

            # Evaluate accuracy on train and val data every epoch
            # vgg_conv.evaluate(sess, train_init_op, accuracy_op, is_train=True)
            # vgg_conv.evaluate(sess, val_init_op, accuracy_op, is_train=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data/coco_data/coco-animals',
                        help='Path to input data directory')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='models/imagenet_vgg16',
                        help='Path to checkpoint directory')
    parser.add_argument('--log_dir', type=str,
                        default='logs/conv_coco',
                        help='Path to log directory')
    parser.add_argument('--model_dir', type=str,
                        default='models/conv_coco',
                        help='Path to model directory')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps to run trainer')
    parser.add_argument('--fc8_steps', type=int, default=5,
                        help='Number of steps to run trainer on last layer')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
