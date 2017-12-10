import os
import sys
import pyprind
import tensorflow as tf
import yaml
import argparse

from src.common import consts
from src.data_preparation import dataset
from src.models import denseNN
from src.common import paths


def train_dev_split(sess_, tf_records_paths_, dev_set_size=2000, batch_size=64, train_sample_size=2000):
    ds_, filenames_ = dataset.features_dataset()

    ds = ds_.shuffle(buffer_size=20000)

    train_ds = ds.skip(dev_set_size).repeat()
    train_ds_iter = train_ds.shuffle(buffer_size=20000) \
        .batch(batch_size) \
        .make_initializable_iterator()

    train_sample_ds = ds.skip(dev_set_size)
    train_sample_ds_iter = train_sample_ds.shuffle(buffer_size=20000) \
        .take(train_sample_size) \
        .batch(train_sample_size) \
        .make_initializable_iterator()

    dev_ds_iter = ds.take(dev_set_size).batch(dev_set_size).make_initializable_iterator()

    sess_.run(train_ds_iter.initializer, feed_dict={filenames_: tf_records_paths_})
    sess_.run(dev_ds_iter.initializer, feed_dict={filenames_: tf_records_paths_})
    sess_.run(train_sample_ds_iter.initializer, feed_dict={filenames_: tf_records_paths_})

    return train_ds_iter.get_next(), dev_ds_iter.get_next(), train_sample_ds_iter.get_next()


def accuracy(x_, output_probs_, name):
    expected = tf.placeholder(tf.int32, shape=([None]), name='expected')
    # exp_vs_output = tf.equal(tf.argmax(output_probs, axis=0), tf.argmax(expected, axis=0))
    exp_vs_output = tf.equal(tf.argmax(output_probs_, 1, output_type=tf.int32), expected)
    accuracy_ = tf.reduce_mean(tf.cast(exp_vs_output, dtype=tf.float32))
    summaries = [tf.summary.scalar(name, accuracy_)]

    merged_summaries = tf.summary.merge(summaries)

    def run(sess_, features, expected_):
        acc, summary_acc = sess_.run([accuracy_, merged_summaries],
                                     feed_dict={x_: features, expected: expected_})

        return acc, summary_acc

    return run


def make_model_name(prefix, batch_size, learning_rate):
    return '%s_%d_%s' % (prefix, batch_size, str(learning_rate).replace('0.', ''))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c', dest="config_filename", type=str, required=False, help='the config file name')
    args = parser.parse_args()

    if args.config_filename:
        with open(args.config_filename, 'r') as yml_file:
            cfg = yaml.load(yml_file)

        BATCH_SIZE = cfg["TRAIN"]["BATCH_SIZE"]
        EPOCHS_COUNT = cfg["TRAIN"]["EPOCHS_COUNT"]
        LEARNING_RATE = cfg["TRAIN"]["LEARNING_RATE"]
        TRAIN_TF_RECORDS = cfg["TRAIN"]["TRAIN_TF_RECORDS"]

        MODEL_NAME = cfg["MODEL"]["MODEL_NAME"]
        MODEL_LAYERS = cfg["MODEL"]["MODEL_LAYERS"]
    else:
        BATCH_SIZE = 128
        EPOCHS_COUNT = 50000
        LEARNING_RATE = 0.0001
        TRAIN_TF_RECORDS = paths.TRAIN_TF_RECORDS

        MODEL_NAME = consts.CURRENT_MODEL_NAME
        MODEL_LAYERS = consts.HEAD_MODEL_LAYERS

    print(MODEL_LAYERS[0])

    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
        next_train_batch, get_dev_ds, get_train_sample_ds = \
            train_dev_split(sess, [TRAIN_TF_RECORDS],
                            dev_set_size=consts.DEV_SET_SIZE,
                            batch_size=BATCH_SIZE,
                            train_sample_size=consts.TRAIN_SAMPLE_SIZE)

        dev_set = sess.run(get_dev_ds)
        dev_set_inception_feature = dev_set[consts.INCEPTION_OUTPUT_FIELD]
        dev_set_y_one_hot = dev_set[consts.LABEL_ONE_HOT_FIELD]

        train_sample = sess.run(get_train_sample_ds)
        train_sample_inception_feature = train_sample[consts.INCEPTION_OUTPUT_FIELD]
        train_sample_y_one_hot = train_sample[consts.LABEL_ONE_HOT_FIELD]

        # x = tf.placeholder(dtype=tf.float32, shape=(None, consts.INCEPTION_CLASSES_COUNT), name="x")
        cost, output_probs, x, y, nn_summaries = denseNN.dense_neural_network(
            consts.HEAD_MODEL_LAYERS, gamma=0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

        train_accu_eval = accuracy(x, output_probs, name='train_accuracy')
        dev_accu_eval = accuracy(x, output_probs, name='test_accuracy')

        nn_merged_summaries = tf.summary.merge(nn_summaries)
        tf.global_variables_initializer().run()

        writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR, MODEL_NAME))

        bar = pyprind.ProgBar(EPOCHS_COUNT, update_interval=1, width=60)

        saver = tf.train.Saver()

        for epoch in range(0, EPOCHS_COUNT):
            batch_features = sess.run(next_train_batch)
            batch_inception_output = batch_features[consts.INCEPTION_OUTPUT_FIELD]
            batch_y = batch_features[consts.LABEL_ONE_HOT_FIELD]

            _, summary = sess.run([optimizer, nn_merged_summaries], feed_dict={
                                      x: batch_inception_output,
                                      y: batch_y
                                  })

            writer.add_summary(summary, epoch)

            _, dev_summaries = dev_accu_eval(sess, dev_set_inception_feature, dev_set_y_one_hot)
            writer.add_summary(dev_summaries, epoch)

            _, train_sample_summaries = train_accu_eval(sess, train_sample_inception_feature, train_sample_y_one_hot)
            writer.add_summary(train_sample_summaries, epoch)

            writer.flush()

            if epoch % 10 == 0 or epoch == EPOCHS_COUNT:
                saver.save(sess, os.path.join(paths.CHECKPOINTS_DIR, MODEL_NAME), latest_filename=MODEL_NAME + '_latest')

            bar.update()
