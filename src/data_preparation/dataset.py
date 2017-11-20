import pandas as pd
import numpy as np
from sklearn import preprocessing
from src.common import paths
import tensorflow as tf
from src.common import consts
from tf_record_utils import read_record_to_queue


def read_record_to_queue(tf_record_name, shapes, preproc_func=None, num_epochs=10, batch_size=32,
                         capacity=2000, min_after_dequeue=1000):
    # this function return images_batch and labels_batch op that can be executed using sess.run
    filename_queue = tf.train.string_input_producer([tf_record_name])  # , num_epochs=num_epochs)

    def read_and_decode_single_example(filename_queue_):
        # first construct a queue containing a list of filenames.
        # this lets a user split up there dataset in multiple files to keep
        # size down

        # Unlike the TFRecordWriter, the TFRecordReader is symbolic
        opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        reader = tf.TFRecordReader(options=opts)
        # One can read a single serialized example from a filename
        # serialized_example is a Tensor of type string.
        _, serialized_example = reader.read(filename_queue_)
        # The serialized example is converted back to actual values.
        # One needs to describe the format of the objects to be returned
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'product_id': tf.FixedLenFeature([], tf.int64),
                consts.IMAGE_RAW_FIELD: tf.FixedLenFeature([], tf.string),
                consts.INCEPTION_OUTPUT_FIELD: tf.FixedLenFeature([consts.INCEPTION_CLASSES_COUNT], tf.float32),
                consts.LABEL_ONE_HOT_FIELD: tf.FixedLenFeature([], tf.int64)
            })

        image_raw = tf.decode_raw(features['img_raw'], tf.uint8)
        label = tf.cast(features[consts.LABEL_ONE_HOT_FIELD], tf.int64)

        image = tf.reshape(image_raw, shapes)
        preproc_image = preproc_func(image) if preproc_func is not None else image
        return label, preproc_image

    # returns symbolic label and image
    label, image = read_and_decode_single_example(filename_queue)

    # groups examples into batches randomly
    images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return images_batch, labels_batch


def validate_train_data_queue(tfrecords_filename, shapes):

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


def read_train_tf_record(record):
    _features = tf.parse_single_example(
        record,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'product_id': tf.FixedLenFeature([], tf.int64),
            consts.IMAGE_RAW_FIELD: tf.FixedLenFeature([], tf.string),
            consts.INCEPTION_OUTPUT_FIELD: tf.FixedLenFeature([consts.INCEPTION_CLASSES_COUNT], tf.float32),
            consts.LABEL_ONE_HOT_FIELD: tf.FixedLenFeature([], tf.int64)
        })
    return _features


def features_dataset():
    file_names = tf.placeholder(tf.string)
    _ds = tf.contrib.data.TFRecordDataset(file_names, compression_type='ZLIB').map(read_train_tf_record)

    return _ds, file_names


def one_hot_label_encoder(csv_path=paths.CATEGORIES):
    category_labels = pd.read_csv(csv_path, dtype={'category_id': np.str})
    lb = preprocessing.LabelBinarizer()
    lb.fit(category_labels['category_id'])

    def find_max_idx(lb_vec):
        lb_vector = lb_vec.reshape(-1).tolist()
        return lb_vector.index(max(lb_vector))

    def encode(lbs_str):
        lbs_vector = np.asarray(lb.transform(lbs_str), dtype=np.float32)
        return np.apply_along_axis(find_max_idx, 1, lbs_vector)

    def decode(one_hots):
        return np.asarray(lb.inverse_transform(one_hots), dtype=np.str)

    return encode, decode


if __name__ == '__main__':
    # one_hot_encoder, _ = one_hot_label_encoder("data/category_names.csv")
    # lb_idx = one_hot_encoder(["1000012764"])
    #
    # print(lb_idx)

    with tf.Graph().as_default() as g, tf.Session().as_default() as sess:
        ds, filenames = features_dataset()
        ds_iter = ds.shuffle(buffer_size=1000, seed=1).batch(10).make_initializable_iterator()
        next_record = ds_iter.get_next()

        sess.run(ds_iter.initializer, feed_dict={filenames: "/data/data/train_example.tfrecords"})
        features = sess.run(next_record)

        # _, one_hot_decoder = one_hot_label_encoder()

        # print(features[consts.INCEPTION_OUTPUT_FIELD])
        print(features[consts.LABEL_ONE_HOT_FIELD])
        print(features['inception_output'].shape)
    # validate_train_data("/data/data/train_example.tfrecords", np.asarray([180, 180, 3]))
