#!/usr/bin/python
import dataset
import bson
import io
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.data import imread
from src.freezing import inception
from src.common import consts


# helper functions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_bson_2_record(input_bson_filename, output_tfrecords_filename, n=None, inception_feature=False):
    one_hot_encoder, _ = dataset.one_hot_label_encoder(csv_path="data/category_names.csv")

    inception_graph = tf.Graph()
    inception_sess = tf.Session(graph=inception_graph)

    with inception_graph.as_default(), inception_sess.as_default():
        inception_model = inception.inception_model()

    z = 0
    data = bson.decode_file_iter(open(input_bson_filename, 'rb'))
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    def get_inception_ouput(img):
        with inception_graph.as_default():
            inception_output = inception_model(inception_sess, img).reshape(-1).tolist()
        return inception_output

    with tf.python_io.TFRecordWriter(output_tfrecords_filename,  options=opts) as writer:
        for c, d in tqdm(enumerate(data), total=n):
            n_img = len(d['imgs'])
            for index in range(n_img):
                img_raw = d['imgs'][index]['picture']
                img = np.array(imread(io.BytesIO(img_raw)))
                # height = img.shape[0]
                # width = img.shape[1]
                product_id = d['_id']
                _feature = {
                    'product_id': _int64_feature(product_id),
                    consts.IMAGE_RAW_FIELD: _bytes_feature(img.tostring())
                }
                if inception_feature:
                    _feature[consts.INCEPTION_OUTPUT_FIELD] = _float_feature(get_inception_ouput(img_raw))
                if 'category_id' in d:
                    _feature[consts.LABEL_ONE_HOT_FIELD] = _int64_feature(int(one_hot_encoder([str(d['category_id'])])[0]))
                example = tf.train.Example(features=tf.train.Features(feature=_feature))
                writer.write(example.SerializeToString())

            z = z + 1
            if (n is not None) and (z % n == 0):
                print(z)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', dest="bson_filename", type=str, required=True, help='the input file in bson format')
    parser.add_argument('-o', dest="tfrecord_filename", type=str, required=True, help='the output file in tfrecrods format')
    parser.add_argument('-n', dest="total_records", type=int, required=False, help='number of records to convert.')
    args = parser.parse_args()

    convert_bson_2_record(args.bson_filename, args.tfrecord_filename, inception_feature=True, n=args.total_records)
