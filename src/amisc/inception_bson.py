from src.freezing import inception
from src.freezing.inception import NodeLookup
import tensorflow as tf
import bson
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    input_bson_filename = "/data/data/train_example.bson"

    node_lookup = NodeLookup()

    inception_graph = tf.Graph()
    inception_sess = tf.Session(graph=inception_graph)

    with inception_graph.as_default(), inception_sess.as_default() as sess:
        inception_model = inception.inception_inference()

        z = 0
        n = 82
        data = bson.decode_file_iter(open(input_bson_filename, 'rb'))
        opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

        with open('inception_feature.txt', 'w') as f:
            for c, d in tqdm(enumerate(data), total=n):
                n_img = len(d['imgs'])
                for index in range(n_img):
                    img_raw = d['imgs'][index]['picture']
                    # height = img.shape[0]
                    # width = img.shape[1]
                    product_id = d['_id']

                    prediction = inception_model(sess, img_raw)
                    predictions = np.squeeze(prediction)
                    top_5 = predictions.argsort()[-5:][::-1]

                    f.write(str(d["_id"]) + " " + node_lookup.id_to_string(top_5[0]) + "\n")
