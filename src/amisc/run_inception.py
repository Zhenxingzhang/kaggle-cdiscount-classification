# Code modified from https://github.com/tensorflow/cleverhans/blob/master/examples/nips17_adversarial_competition/sample_defenses/base_inception_model/defense.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from PIL import Image
from src.amisc.ReadableLabel import ReadableLabel

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path',
    '/Users/zhenxingzhang/Documents/2017/kaggle-cdiscount-classification/data/inception-v3/',
    'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', 'data/inception-imgs', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size_', 2, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size_ = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.jpg'))[:20]:
        with tf.gfile.Open(filepath, "rb") as f:
            image = Image.open(f)#, mode='RGB').astype(np.float) / 255.0
            image = image.resize((299, 299), Image.ANTIALIAS)
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = np.array(image) / 255.0 * 2.0 - 1.0#eval_image(image, 299, 299)
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size_:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


if __name__ == '__main__':
    batch_shape = [FLAGS.batch_size_, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001
    predictions = []

    tf.logging.set_verbosity(tf.logging.INFO)

    node_lookup = ReadableLabel("data/inception-v3/imagenet1000_clsid_to_human.txt")

    with tf.Graph().as_default():
        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
              x_input, num_classes=num_classes, is_training=False, dropout_keep_prob=1.0)
            variables_to_restore = slim.get_variables_to_restore()

        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        # saver = tf.train.Saver(tf.global_variables())
        # Start session
        with tf.Session() as sess:

            # Restore previously trained variables from disk
            # Variables constructed in forward_pass() will be initialised with
            # values restored from variables with the same name
            # Note: variable names MUST match for it to work
            print("Restoring Saved Variables from Checkpoint: {}"
                  .format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
        #
        # session_creator = tf.train.ChiefSessionCreator(
        #     scaffold=tf.train.Scaffold(saver=saver),
        #     checkpoint_filename_with_path=FLAGS.checkpoint_path,
        #     master=FLAGS.master)
        #
        # with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                labels = sess.run(predicted_labels, feed_dict={x_input: images})

                for filename, label in zip(filenames, labels):
                    # true_label = image_labels.merge(pd.DataFrame({"ImageId":[filename[:-4]]}), on="ImageId")["TrueLabel"][0]
                    # predictions.append([filename[:-4], label])
                    print(filename, label, node_lookup.get_human_readable(label))

    # pd.DataFrame(predictions, columns=["ImageId", "PredictedLabel"])#.to_csv("predictions.csv")