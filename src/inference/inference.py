import os

import numpy as np
import pandas as pd
import tensorflow as tf
import csv
from src.common import consts
from src.data_preparation import dataset
from src.models import denseNN
from src.common import paths
import argparse
import yaml
from src.training.train_model import neural_model


def infer_test(model_name, x_, output_probs_, batch_size, test_tfrecords_files, test_prediction_csv):

    _, one_hot_decoder = dataset.one_hot_label_encoder(csv_path="data/category_names.csv")

    with tf.Session().as_default() as sess:
        ds, filename = dataset.test_features_dataset()
        ds_iter = ds.batch(batch_size).make_initializable_iterator()
        sess.run(ds_iter.initializer, feed_dict={filename: test_tfrecords_files})

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        lines = open(os.path.join(paths.CHECKPOINTS_DIR, model_name, str(LEARNING_RATE),  model_name + '_latest')).read().split('\n')
        latest_checkpoint = [l.split(':')[1].replace('"', '').strip() for l in lines if 'model_checkpoint_path:' in l][0]
        check_point_path = os.path.join(paths.CHECKPOINTS_DIR, model_name, str(LEARNING_RATE), latest_checkpoint)
        print("Restore from {}".format(check_point_path))
        saver.restore(sess, check_point_path)

        with open(test_prediction_csv, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["_id", "category_id", "prob"])
            try:
                while True:
                    test_batch = sess.run(ds_iter.get_next())

                    ids = test_batch['_id']
                    inception_features = test_batch[consts.INCEPTION_OUTPUT_FIELD]

                    print(ids.shape)
                    probabilities_ = sess.run(output_probs_, feed_dict={x_: inception_features})
                    pred_labels = one_hot_decoder(probabilities_)
                    max_probs = np.apply_along_axis(np.amax, 1, probabilities_)

                    for (_id, pred_label, prob) in zip(ids, pred_labels, max_probs):
                        csv_writer.writerow([_id, pred_label, prob])

                    # test_df = pd.DataFrame(data=zip(ids, pred_labels, max_probs), columns=["_id", "category_id", "prob"])
                    #
                    # idx = test_df.groupby(['_id'])['prob'].transform(max) == test_df['prob']
                    #
                    # test_df = test_df[idx]
                    #
                    # for index, row in test_df.iterrows():
                    #     csv_writer.writerow([row._id, row.category_id, row.prob])

            except tf.errors.OutOfRangeError:
                print('End of the dataset')

            # agg_test_df.to_csv(paths.TEST_PREDICTIONS, index_label='id', float_format='%.17f')

            print('predictions saved to %s' % test_prediction_csv)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c', dest="config_filename", type=str, required=True, help='the config file name')
    parser.add_argument('-i', dest="tfrecord_filename", type=str, required=False, help='tfrecord file')
    parser.add_argument('-o', dest="csv_filename", type=str, required=False, help='csv')
    args = parser.parse_args()

    with open(args.config_filename, 'r') as yml_file:
        cfg = yaml.load(yml_file)

    BATCH_SIZE = cfg["TEST"]["BATCH_SIZE"]
    if args.tfrecord_filename is not None:
        TEST_TF_RECORDS = args.tfrecord_filename
    else:
        TEST_TF_RECORDS = cfg["TEST"]["TEST_TF_RECORDS"]

    if args.csv_filename is not None:
        TEST_OUTPUT_CSV = args.csv_filename
    else:
        TEST_OUTPUT_CSV = cfg["TEST"]["OUTPUT_CSV_PATH"]

    MODEL_NAME = cfg["MODEL"]["MODEL_NAME"]
    MODEL_LAYERS = cfg["MODEL"]["MODEL_LAYERS"]
    LEARNING_RATE = cfg["TRAIN"]["LEARNING_RATE"]

    print("Testing model name: {}".format(MODEL_NAME))
    print("Testing data: {}".format(TEST_TF_RECORDS))
    print("Testing output: {}".format(TEST_OUTPUT_CSV))

    with tf.Graph().as_default():
        # x = tf.placeholder(dtype=tf.float32, shape=(consts.INCEPTION_CLASSES_COUNT, None), name="x")
        # _, output_probs, x, _, _ = denseNN.dense_neural_network(consts.HEAD_MODEL_LAYERS, gamma=0.01)
        x, _, output_ = neural_model(MODEL_LAYERS)
        output_probs = tf.nn.softmax(output_)

        print(output_probs.shape)
        infer_test(MODEL_NAME, x, output_probs, BATCH_SIZE, TEST_TF_RECORDS, TEST_OUTPUT_CSV)
