import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import pyprind
import os
from src.common import paths

if __name__ == '__main__':
    batch_shape = [64, 299, 299, 3]
    num_classes = 5270
    predictions = []
    checkpoint_path = "/data/inception/2016/"
    LEARNING_RATE = 0.0001
    EPOCHS_COUNT = 5000
    MODEL_NAME = "inception_v3_cdiscount"

    slim = tf.contrib.slim

    with tf.Graph().as_default(), tf.Session().as_default() as sess:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)

        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        y = tf.placeholder(dtype=tf.int32, shape=(None), name="y")

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            y_, end_points = inception.inception_v3(
              x_input, num_classes=num_classes, is_training=True, dropout_keep_prob=0.8)
        print(y_.shape)

        exclude = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        with tf.name_scope('cross_entropy'):
            cross_entropy_i = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_)
            cross_entropy = tf.reduce_mean(cross_entropy_i)
            tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_, 1, output_type=tf.int32), y)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        # training step (NOTE: improved optimiser and lower learning rate; needed for more complex model)
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        # Merge all the summaries and write them out to /summaries/conv (by default)
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR, MODEL_NAME, str(LEARNING_RATE), 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(paths.SUMMARY_DIR, MODEL_NAME, str(LEARNING_RATE), 'test'))

        # sess.run(tf.global_variables_initializer()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(variables_to_restore)

        # Restore previously trained variables from disk
        # Variables constructed in forward_pass() will be initialised with
        # values restored from variables with the same name
        # Note: variable names MUST match for it to work
        print("Restoring Saved Variables from Checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

        bar = pyprind.ProgBar(EPOCHS_COUNT, update_interval=1, width=60)
        for epoch in range(EPOCHS_COUNT):
            bar.update()

