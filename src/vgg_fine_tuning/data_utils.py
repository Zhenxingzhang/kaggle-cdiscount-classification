"""
Helper functions for data processing
"""

from functools import partial
import tensorflow as tf
import vgg_preprocessing


def _decode_jpeg(image_string):
    '''
    Decode image string Tensor to 3D float Tensor
    '''
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    return tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)


def _parse_features(serialised_proto):
    '''
    Decode proto to features
    Return 3D image tensor and label
    '''
    features = tf.parse_single_example(
        serialised_proto,
        features={
            # shape not required as all values are rank 0
            '_id': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label_one_hot': tf.FixedLenFeature([], tf.int64)
        })
    # Decode string image to 3D float Tensor
    image = _decode_jpeg(features['img_raw'])
    label = features['label_one_hot']

    return image, label


def preprocess_and_batch_data(filename, no_threads, batch_size, is_training):
    '''
    Input tfrecords from files
    Preprocess images in parallel
    Shuffle and batch data
    '''
    # Create TFRecordDataset from filenames tensor containing files where tfrecords are stored
    dataset = tf.contrib.data.TFRecordDataset(filename, compression_type='ZLIB')

    # Parse record dataset into tensors
    dataset = dataset.map(_parse_features)

    # Preprocess images with multiple threads in parallel
    processor = vgg_preprocessing.preprocess_data
    partial_preprocess_data = partial(processor, is_training=is_training)
    preprocessed_data = dataset.map(partial_preprocess_data,
                                    num_threads=no_threads,
                                    output_buffer_size=batch_size)

    # Shuffle data
    shuffled_data = preprocessed_data.shuffle(buffer_size=10000)
    return shuffled_data.batch(batch_size)