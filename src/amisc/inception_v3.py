"""
inception v3 checkpoint file: https://github.com/tensorflow/models/tree/master/research/slim
human readable labels: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
"""
import os
import numpy as np
from scipy.misc import imread
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from src.amisc.ReadableLabel import ReadableLabel


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.jpg')):
        with tf.gfile.Open(filepath) as f:
          image = imread(f, mode='RGB').astype(np.float)
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image / 255.0 * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
          yield filenames, images
          filenames = []
          images = np.zeros(batch_shape)
          idx = 0
    if idx > 0:
        yield filenames, images


def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    node_lookup = ReadableLabel("data/inception-v3/imagenet1000_clsid_to_human.txt")

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
          _, end_points = inception.inception_v3(
              x_input, num_classes=num_classes, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        inception_feature = end_points['PreLogits']

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.checkpoint_path,
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
          # with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
              labels, features = sess.run([predicted_labels, inception_feature], feed_dict={x_input: images})
              print(features.shape)
              for filename, label in zip(filenames, labels):
                # out_file.write('{0},{1}\n'.format(filename, label))
                print(filename, label, node_lookup.get_human_readable(label))


if __name__ == '__main__':
    slim = tf.contrib.slim

    tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')
    tf.flags.DEFINE_string(
        'checkpoint_path', 'data/inception-v3/inception_v3.ckpt', 'Path to checkpoint for inception network.')
    tf.flags.DEFINE_string(
        'input_dir', 'data/inception-imgs', 'Input directory with images.')
    tf.flags.DEFINE_string(
        'output_file', '', 'Output file to save labels.')
    tf.flags.DEFINE_integer(
        'image_width', 299, 'Width of each input images.')
    tf.flags.DEFINE_integer(
        'image_height', 299, 'Height of each input images.')
    tf.flags.DEFINE_integer(
        'batch_size', 3, 'How many images process at one time.')

    FLAGS = tf.flags.FLAGS
    tf.app.run()
