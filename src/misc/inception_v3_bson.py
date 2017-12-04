from src.misc.ReadableLabel import ReadableLabel
from tensorflow.contrib.slim.nets import inception
import tensorflow as tf
import bson
import io
from PIL import Image
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    input_bson_filename = "/data/data/train_example.bson"
    checkpoint_path = "/Users/zhenxingzhang/Documents/2017/kaggle-cdiscount-classification/data/inception-v3/"
    node_lookup = ReadableLabel("data/inception-v3/imagenet1000_clsid_to_human.txt")
    inception_graph = tf.Graph()
    inception_sess = tf.Session(graph=inception_graph)

    slim = tf.contrib.slim
    batch_shape = [1, 299, 299, 3]
    num_classes = 1001
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)

    with inception_graph.as_default(), inception_sess.as_default() as sess:
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=num_classes, is_training=False, dropout_keep_prob=1.0)

        inception_feature = end_points['PreLogits']
        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        # Run computation
        print("Restoring Saved Variables from Checkpoint: {}".format(latest_checkpoint))
        saver = tf.train.Saver(slim.get_model_variables())
        saver.restore(sess, latest_checkpoint)

        z = 0
        n = 82
        data = bson.decode_file_iter(open(input_bson_filename, 'rb'))

        with open('inception_feature_batch.txt', 'w') as f:
            for c, d in tqdm(enumerate(data), total=n):
                n_img = len(d['imgs'])
                for index in range(n_img):
                    img_raw = d['imgs'][index]['picture']
                    img = Image.open(io.BytesIO(img_raw))
                    img = img.resize((299, 299), Image.ANTIALIAS)
                    img = np.array(img)
                    images = np.zeros([1, 299, 299, 3], float)
                    images[0, :] = img / 255.0 * 2.0 - 1.0

                    product_id = d['_id']

                    labels, features = sess.run([predicted_labels, inception_feature], feed_dict={x_input: images})
                    f.write(str(d["_id"]) + " " + node_lookup.get_human_readable(labels[0]) + "\n")

                z = z + 1
                if (n is not None) and (z % n == 0):
                    print(z)
                    break
