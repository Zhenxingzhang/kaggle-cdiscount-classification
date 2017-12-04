# https://stackoverflow.com/questions/43092439/shuffling-and-training-in-tensorflow
# https://stackoverflow.com/questions/44132307/tf-contrib-data-dataset-repeat-with-shuffle-notice-epoch-end-mixed-epochs

import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    data = np.array([1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4, 5, 5, 5, 5])
    # data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    #https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
    #buffer_size  Instead of shuffling the entire dataset, it maintains a buffer of buffer_size elements, and randomly selects the next element from that buffer
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=10).skip(10).batch(2)
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(one_element))
        except tf.errors.OutOfRangeError:
            print("end!")
