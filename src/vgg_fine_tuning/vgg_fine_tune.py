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
from src.data_preparation import dataset
from src.common import consts
import numpy as np
from src.vgg_fine_tuning.vgg_preprocessing import _decode_jpeg
from skimage.data import imread
import io
from PIL import Image


def get_data_iter(sess_, tf_records_paths_, buffer_size=200, batch_size=10):
    ds_, file_names_ = dataset.image_dataset()
    ds_iter = ds_.shuffle(buffer_size).repeat().batch(batch_size).make_initializable_iterator()
    # return ds_iter
    sess_.run(ds_iter.initializer, feed_dict={file_names_: tf_records_paths_})
    return ds_iter, file_names_, ds_iter.get_next()


def decode_img(img_raw_):
    # img_ = np.array(imread(io.BytesIO(img_raw_)))
    # return img_ / 255.0 * 2.0 - 1.0
    from PIL import Image
    import StringIO
    tempBuff = StringIO.StringIO()
    tempBuff.write(img_raw_)
    tempBuff.seek(0)  # need to jump back to the beginning before handing it off to PIL

    return np.array(Image.open(tempBuff).resize((224, 224), Image.ANTIALIAS))


# Silence compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = None
NO_CLASSES = 5270
NO_THREADS = 4
BATCH_SIZE = 10
EVAL_BATCH_SIZE = 300
DROPOUT_KEEP_PROB = 1.0
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 0.001
MOVING_AVERAGE_DECAY = 0.9999
LEARNING_RATE_DECAY_FACTOR = 0.8


def main(_):

    graph = tf.Graph()
    with graph.as_default():

        # # Preprocess and batch train data
        # batched_train_data = data_utils.preprocess_and_batch_data(
        #     train_files, train_labs, NO_THREADS, BATCH_SIZE)
        # # Preprocess and batch val data
        # batched_val_data = data_utils.preprocess_and_batch_data(
        #     val_files, val_labs, NO_THREADS, BATCH_SIZE)

        # Define an iterator that can operator on either dataset
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images/labels
        # as they are automatically pulled out from the iterator queues

        # A reinitializable iterator is defined by its structure
        # We could use the `output_types` and `output_shapes` properties of
        # either `train_data` or `val_data` here because they are compatible
        # iterator = tf.contrib.data.Iterator.from_structure(
        #     batched_train_data.output_types,
        #     batched_train_data.output_shapes)
        # images, labels = iterator.get_next()
        #
        # train_init_op = iterator.make_initializer(batched_train_data)
        # val_init_op = iterator.make_initializer(batched_val_data)

        # Prepare graph
        images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        labels = tf.placeholder(dtype=tf.int32, shape=(None), name="y")

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

        with tf.name_scope('input_images'):
            tf.summary.image('input', images, 1)

    with tf.Session(graph=graph) as sess:
        training_files ="/data/data/train_example_images.tfrecord"
        eval_files = "/data/data/train_example_images.tfrecord"
        iter, train_files_ops, next_train_batch = get_data_iter(sess, training_files, batch_size=BATCH_SIZE)
        _, _, next_eval_batch = get_data_iter(sess, eval_files, batch_size=EVAL_BATCH_SIZE)

        # image_strings = tf.placeholder(tf.string, shape=[None])
        # inputs_images = image_strings.map(_decode_jpeg)

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
            # sess.run(train_init_op)
            sess.run(iter.initializer, feed_dict={train_files_ops: training_files})

            count = 0
            while True:
                try:
                    batch_examples = sess.run(next_train_batch)
                    batch_images_raw = batch_examples[consts.IMAGE_RAW_FIELD]
                    # batch_images = np.array(map(decode_img, batch_images_raw))
                    batch_images = batch_examples["_image"]
                    # print(batch_images.shape)
                    batch_y = batch_examples[consts.LABEL_ONE_HOT_FIELD]
                    # print(batch_y)
                    # Train and write summaries
                    t0 = time.time()
                    summary, _, los = sess.run(
                        [summary_op, fc8_train_op, loss_op],
                        feed_dict={images: batch_images,
                                   labels: batch_y})

                    summary_writer.add_summary(summary, count)
                    print(("Step: {}, Loss: {}, [timer: {:.2f}s]")
                          .format(count, los, time.time() - t0))
                    count += 1

                    if count % 10 == 0:
                        eval_batch_examples = sess.run(next_eval_batch)
                        eval_batch_images = eval_batch_examples["_image"]
                        eval_batch_y = eval_batch_examples[consts.LABEL_ONE_HOT_FIELD]

                        [eval_summaries, eval_loss] = sess.run([summary_op, loss_op], feed_dict={
                            images: eval_batch_images,
                            labels: eval_batch_y})

                        print(("Step: {}, Loss: {}, [timer: {:.2f}s]")
                              .format(count, eval_loss, time.time() - t0))

                except tf.errors.OutOfRangeError:
                    break

            # Evaluate accuracy on train and val data every epoch
            # vgg_conv.evaluate(sess, train_init_op, accuracy_op, is_train=True)
            # vgg_conv.evaluate(sess, val_init_op, accuracy_op, is_train=False)

        # Train full model continuing with the same weights
        print("Training Full Model:")
        for epoch in range(FLAGS.steps):
            print("Epoch {}/{}".format(epoch + 1, FLAGS.steps))
            sess.run(iter.initializer, feed_dict={train_files_ops: training_files})
            count = 0
            while True:
                try:
                    # Train and write summaries
                    t0 = time.time()
                    batch_examples = sess.run(next_train_batch)
                    batch_images = batch_examples[consts.IMAGE_RAW_FIELD]
                    print(batch_images.shape)
                    batch_y = batch_examples[consts.LABEL_ONE_HOT_FIELD]

                    summary, _, los = sess.run(
                        [summary_op, full_train_op, loss_op],
                        feed_dict={images: batch_images,
                                   labels: batch_y})

                    summary_writer.add_summary(summary, count)
                    # print(("Step: {}, Loss: {}, [timer: {:.2f}s]")
                    #       .format(count, los, time.time() - t0))
                    count += 1

                    if count % 10 == 0:
                        eval_batch_examples = sess.run(next_eval_batch)
                        eval_batch_images = eval_batch_examples["_image"]
                        eval_batch_y = eval_batch_examples[consts.LABEL_ONE_HOT_FIELD]

                        [eval_summaries, eval_loss] = sess.run([summary_op, loss_op], feed_dict={
                            images: eval_batch_images,
                            labels: eval_batch_y})

                        print(("Step: {}, Loss: {}, [timer: {:.2f}s]")
                              .format(count, eval_loss, time.time() - t0))
                        # test_writer.add_summary(eval_summaries, epoch)
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
                        default='/data/imagenet_vgg16/',
                        help='Path to checkpoint directory')
    parser.add_argument('--log_dir', type=str,
                        default='/data/summary/Fine-Tune-Vgg',
                        help='Path to log directory')
    parser.add_argument('--model_dir', type=str,
                        default='models/conv_coco',
                        help='Path to model directory')
    parser.add_argument('--steps', type=int, default=0,
                        help='Number of steps to run trainer')
    parser.add_argument('--fc8_steps', type=int, default=10,
                        help='Number of steps to run trainer on last layer')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
