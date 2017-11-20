import tensorflow as tf
from src.common import consts

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_int64_feature(example, name):
    return int(example.features.feature[name].int64_list.value[0])


def get_float_feature(example, name):
    return int(example.features.feature[name].float_list.value)


def get_bytes_feature(example, name):
    return example.features.feature[name].bytes_list.value[0]


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
                consts.INCEPTION_OUTPUT_FIELD: tf.FixedLenFeature([], tf.float32),
                consts.LABEL_ONE_HOT_FIELD: tf.FixedLenFeature([], tf.int64)
            })

        image_raw = tf.decode_raw(features['img_raw'], tf.uint8)
        label = tf.cast(features['category_id'], tf.int64)

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
