#!/usr/bin/python
import dataset

import bson
import getopt
import io
import sys
import argparse

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.data import imread


# helper functions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_bson_2_record(input_bson_filename, output_tfrecords_filename, n=10000):

    z = 0
    data = bson.decode_file_iter(open(input_bson_filename, 'rb'))
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    with tf.python_io.TFRecordWriter(output_tfrecords_filename,  options=opts) as writer:
        for c, d in tqdm(enumerate(data), total=n):
            n_img = len(d['imgs'])
            for index in range(n_img):
                img_raw = d['imgs'][index]['picture']
                img = np.array(imread(io.BytesIO(img_raw)))
                height = img.shape[0]
                width = img.shape[1]
                product_id = d['_id']
                _feature = {
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'product_id': _int64_feature(product_id),
                    'img_raw': _bytes_feature(img.tostring())
                }
                if 'category_id' in d:
                    _feature['category_id'] = _int64_feature(d['category_id'])
                example = tf.train.Example(features=tf.train.Features(feature=_feature))
                writer.write(example.SerializeToString())

            z = z + 1
            if z % n == 0:
                print(z)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', dest="bson_filename", type=str, required=True, help='the input file in bson format')
    parser.add_argument('-o', dest="tfrecord_filename", type=str, required=True, help='the output file in tfrecrods format')
    parser.add_argument('-n', dest="total_records", type=int, required=True, help='number of records to convert.')
    args = parser.parse_args()

    one_hot_encoder, _ = dataset.one_hot_label_encoder()

    convert_bson_2_record(args.bson_filename, args.tfrecord_filename, args.total_records)
