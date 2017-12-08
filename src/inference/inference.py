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
from src.training.train_model import linear_model


def infer_test(model_name, x_, predict_labels_, batch_size, test_tfrecords_files, test_prediction_csv):

    _, one_hot_decoder = dataset.one_hot_label_encoder(csv_path="data/category_names.csv")

    with tf.Session().as_default() as sess:
        ds, filename = dataset.test_features_dataset()
        ds_iter = ds.batch(batch_size).make_initializable_iterator()
        sess.run(ds_iter.initializer, feed_dict={filename: test_tfrecords_files})

        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        lines = open(os.path.join(paths.CHECKPOINTS_DIR, model_name + '_latest')).read().split('\n')
        latest_checkpoint = [l.split(':')[1].replace('"', '').strip() for l in lines if 'model_checkpoint_path:' in l][0]
        saver.restore(sess, os.path.join(paths.CHECKPOINTS_DIR, latest_checkpoint))
        #
        # category_id = one_hot_decoder(np.identity(consts.CLASSES_COUNT))
        # agg_test_df = pd.DataFrame(columns=["_id", "category_id"])

        with open(test_prediction_csv, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["_id", "category_id"])
            try:
                while True :
                    test_batch = sess.run(ds_iter.get_next())

                    ids = test_batch['_id']
                    inception_features = test_batch[consts.INCEPTION_OUTPUT_FIELD]

                    print(ids.shape)
                    pred_labels = sess.run(predict_labels_, feed_dict={x_: inception_features})
                    #
                    # #print(pred_probs.shape)
                    #
                    # test_df = pd.DataFrame(data=pred_labels, columns=category_id)
                    # test_df.index = ids
                    #
                    # if agg_test_df is None:
                    #     agg_test_df = test_df
                    # else:
                    #     agg_test_df = agg_test_df.append(test_df)
                    for (_id, p_label) in zip(ids, pred_labels):
                        csv_writer.writerow([_id, one_hot_decoder(p_label)[0]])

            except tf.errors.OutOfRangeError:
                print('End of the dataset')

            # agg_test_df.to_csv(paths.TEST_PREDICTIONS, index_label='id', float_format='%.17f')

            print('predictions saved to %s' % test_prediction_csv)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-c', dest="config_filename", type=str, required=True, help='the config file name')
    args = parser.parse_args()

    with open(args.config_filename, 'r') as yml_file:
        cfg = yaml.load(yml_file)

    BATCH_SIZE = cfg["TEST"]["BATCH_SIZE"]
    TEST_TF_RECORDS = cfg["TEST"]["TEST_TF_RECORDS"]
    TEST_OUTPUT_CSV = cfg["TEST"]["OUTPUT_CSV_PATH"]

    MODEL_NAME = cfg["MODEL"]["MODEL_NAME"]
    MODEL_LAYERS = cfg["MODEL"]["MODEL_LAYERS"]

    print("Testing model name: {}".format(MODEL_NAME))
    print("Testing data: {}".format(TEST_TF_RECORDS))
    print("Testing output: {}".format(TEST_OUTPUT_CSV))

    with tf.Graph().as_default():
        # x = tf.placeholder(dtype=tf.float32, shape=(consts.INCEPTION_CLASSES_COUNT, None), name="x")
        # _, output_probs, x, _, _ = denseNN.dense_neural_network(consts.HEAD_MODEL_LAYERS, gamma=0.01)
        x, _, output_probs = linear_model(MODEL_LAYERS)
        predictions = tf.argmax(output_probs, 1)

        print(predictions.shape)
        infer_test(MODEL_NAME, x, predictions, BATCH_SIZE, TEST_TF_RECORDS, TEST_OUTPUT_CSV)
