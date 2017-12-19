"""
Helper functions for data processing
"""

import tensorflow as tf
import vgg_preprocessing
import os


def read_coco_files_and_labels(directory):
    '''
    Iterate through directory structure of COCO dataset
    Numerate the labels
    Return filenames of data with corresponding labels
    '''
    labels = os.listdir(directory)
    files_to_label = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_to_label.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_to_label)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))
    one_hot = {}
    for i, label in enumerate(unique_labels):
        one_hot[label] = i
    labels = [one_hot[l] for l in labels]

    return filenames, labels


def preprocess_and_batch_data(files, labels, no_threads, batch_size):
    '''
    Input data from files with corresponding labels
    Preprocess images in parallel
    Shuffle and batch data
    '''
    # Define input Tensors
    files = tf.constant(files)
    labels = tf.constant(labels)
    # Load files via queues, preprocess images with multiple threads in
    # parallel
    # Use tf.data.Dataset to slice tensors into dataset for feeding data
    # queues into model
    data = tf.contrib.data.Dataset.from_tensor_slices((files, labels))
    preprocessed_data = data.map(vgg_preprocessing.preprocess_data,
                                 num_threads=no_threads,
                                 output_buffer_size=batch_size)
    # Shuffle data
    shuffled_data = preprocessed_data.shuffle(buffer_size=10000)
    return shuffled_data.batch(batch_size)
