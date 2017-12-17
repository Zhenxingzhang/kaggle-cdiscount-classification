import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import pyprind
import os
from src.common import paths
from src.data_preparation import dataset
from src.training.train_model import get_tfrecrods_files
from src.common import consts
from skimage.data import imread
import io
import numpy as np
import argparse
import yaml

def get_data_iter(sess_, tf_records_paths_, buffer_size=20, batch_size=64):
    ds_, file_names_ = dataset.image_dataset()
    ds_iter = ds_.shuffle(buffer_size).repeat().batch(batch_size).make_initializable_iterator()
    sess_.run(ds_iter.initializer, feed_dict={file_names_: tf_records_paths_})
    return ds_iter.get_next()


def decode_img(img_raw_):
    img_ = np.array(imread(io.BytesIO(img_raw_)))
    return img_ / 255.0 * 2.0 - 1.0


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

    print(TRAIN_TF_RECORDS)
    training_files = get_tfrecrods_files(TRAIN_TF_RECORDS)
    eval_files = get_tfrecrods_files(EVAL_TF_RECORDS)

    print("Training model {}".format(MODEL_NAME))
    print("Training data {}".format(training_files))
    print("Training batch size {}".format(BATCH_SIZE))

    print("Evaluation data {}".format(eval_files))
    print("Evaluation batch size {}".format(EVAL_BATCH_SIZE))

    batch_shape = [None, 180, 180, 3]
    num_classes = 5270
    inception_checkpoint_path = "/data/inception/2016/"

    TRAIN_TF_RECORDS = "/data/data/train_example_images.tfrecord"

    checkpoint_path = os.path.join(paths.CHECKPOINTS_DIR, MODEL_NAME, str(LEARNING_RATE), MODEL_NAME)

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        print("Please create checkpoint path: {}".format(checkpoint_path))

    slim = tf.contrib.slim

    with tf.Graph().as_default(), tf.Session().as_default() as sess:

        next_train_batch = get_data_iter(sess, training_files, batch_size=BATCH_SIZE)
        next_eval_batch = get_data_iter(sess, eval_files, batch_size=EVAL_BATCH_SIZE)

        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        y = tf.placeholder(dtype=tf.int32, shape=(None), name="y")

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            y_, end_points = inception.inception_v3(
                x_input, num_classes=num_classes, is_training=True, dropout_keep_prob=1.0)
            # exclude = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
            # variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
            variables_to_save = slim.get_variables_to_restore()

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
        #
        # saver_write = tf.train.Saver(variables_to_save)

        # Restore previously trained variables from disk
        # Variables constructed in forward_pass() will be initialised with
        # values restored from variables with the same name
        # Note: variable names MUST match for it to work
        # print("Restoring Saved Variables from Checkpoint: {}".format(latest_checkpoint))
        # saver.restore(sess, latest_checkpoint)

        bar = pyprind.ProgBar(EPOCHS_COUNT, update_interval=1, width=60)
        for epoch in range(EPOCHS_COUNT):
            batch_examples = sess.run(next_train_batch)
            batch_ids = batch_examples['_id']
            batch_images_raw = batch_examples[consts.IMAGE_RAW_FIELD]
            batch_images = np.array(map(decode_img, batch_images_raw))
            batch_y = batch_examples[consts.LABEL_ONE_HOT_FIELD]

            _, summary = sess.run([optimizer, merged], feed_dict={
                x_input: batch_images,
                y: batch_y
            })

            train_writer.add_summary(summary, epoch)

            if epoch % 100 == 0 or epoch == EPOCHS_COUNT:
                eval_batch_examples = sess.run(next_eval_batch)
                eval_batch_images_raw = eval_batch_examples[consts.IMAGE_RAW_FIELD]
                eval_batch_images = np.array(map(decode_img, eval_batch_images_raw))
                eval_batch_y = eval_batch_examples[consts.LABEL_ONE_HOT_FIELD]

                dev_summaries = sess.run(merged, feed_dict={
                    x_input: eval_batch_images,
                    y: eval_batch_y
                })
                test_writer.add_summary(dev_summaries, epoch)

                saver.save(sess, checkpoint_path, latest_filename=MODEL_NAME + '_latest')

            bar.update()
