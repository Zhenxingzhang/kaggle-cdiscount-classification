#!/usr/bin/python
import dataset
import bson
import io
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from src.common import consts
from tensorflow.contrib.slim.nets import inception
from src.data_preparation.tf_record_utils import float_feature, int64_feature


def convert_bson_2_record(input_bson_filename, output_tfrecords_filename,
                          checkpoint_path="/data/inception/2016/",
                          n=82, batch_size=10):
    one_hot_encoder, _ = dataset.one_hot_label_encoder(csv_path="data/category_names.csv")

    inception_graph = tf.Graph()
    inception_sess = tf.Session(graph=inception_graph)

    slim = tf.contrib.slim

    batch_shape = [None, 299, 299, 3]
    num_classes = 1001
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)

    with inception_graph.as_default(), inception_sess.as_default() as sess:
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input,
                num_classes=num_classes,
                is_training=False,
                dropout_keep_prob=1.0)

        inception_feature = end_points['PreLogits']

        # Run computation
        print("Restoring Saved Variables from Checkpoint: {}".format(latest_checkpoint))
        saver = tf.train.Saver(slim.get_model_variables())
        saver.restore(sess, latest_checkpoint)

        data = bson.decode_file_iter(open(input_bson_filename, 'rb'))
        opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

        batch_images = []
        batch_sample = []
        batch_idx = 0

        with tf.python_io.TFRecordWriter(output_tfrecords_filename,  options=opts) as writer:
            for c, d in tqdm(enumerate(data), total=n):
                n_img = len(d['imgs'])
                for index in range(n_img):
                    img_raw = d['imgs'][index]['picture']
                    img = Image.open(io.BytesIO(img_raw))
                    img = img.resize((299, 299), Image.ANTIALIAS)
                    img = np.array(img)
                    batch_images.append(img / 255.0 * 2.0 - 1.0)

                    sample = dict()
                    sample["_id"] = d['_id']
                    if 'category_id' in d:
                        sample["category_id"] = d['category_id']
                    batch_sample.append(sample)

                batch_idx = batch_idx + 1
                # print("{}".format(batch_idx))
                if batch_idx / batch_size == 1 or c == n-1:
                    # print("{} full fill one batch, batch images: {}".format(c, len(batch_images)))
                    features = sess.run(inception_feature, feed_dict={x_input: np.array(batch_images)})

                    for idx in range(len(batch_sample)):
                        _feature = dict()
                        _feature["_id"] = int64_feature(batch_sample[idx]["_id"])
                        _feature[consts.INCEPTION_OUTPUT_FIELD] = float_feature(np.squeeze(features[idx, :]))
                        if "category_id" in batch_sample[idx]:
                            category_id = batch_sample[idx]["category_id"]
                            _feature[consts.LABEL_ONE_HOT_FIELD] = \
                                int64_feature(int(one_hot_encoder([str(category_id)])[0]))
                        example = tf.train.Example(features=tf.train.Features(feature=_feature))
                        writer.write(example.SerializeToString())
                    batch_idx = 0
                    batch_images = []
                    batch_sample = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', dest="bson_filename", type=str, required=True, help='the input file in bson format')
    parser.add_argument('-o', dest="tfrecord_filename", type=str, required=True, help='the output file in tfrecrods format')
    parser.add_argument('-n', dest="total_records", type=int, required=False, help='number of records to convert.')
    parser.add_argument('-b', dest="batch_size", type=int, required=False, help='size of batch.')
    args = parser.parse_args()

    convert_bson_2_record(args.bson_filename, args.tfrecord_filename, n=args.total_records, batch_size=args.batch_size)
