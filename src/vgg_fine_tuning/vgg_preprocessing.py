"""
Preprocessing for VGG
"""

import tensorflow as tf
from src.common import consts

R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94

CROP_HEIGHT = 224
CROP_WIDTH = 224

RESIZE_SIDE_MIN = 256
RESIZE_SIDE_MAX = 512


def preprocess_tf_data(feature, is_training=False,
                    resize_side_min=RESIZE_SIDE_MIN,
                    resize_side_max=RESIZE_SIDE_MAX,
                    output_height=CROP_HEIGHT,
                    output_width=CROP_WIDTH):
    '''
    Read image from file
    Preprocess image
    Return image and label
    '''
    image = _decode_jpeg(feature[consts.IMAGE_RAW_FIELD])
    label = feature[consts.LABEL_ONE_HOT_FIELD]

    if (is_training):
        return _preprocess_for_train(image, label, output_height,
                                     output_width, resize_side_min,
                                     resize_side_max)
    else:
        return _preprocess_for_val(image, label, output_height,
                                   output_width, resize_side_min)


def preprocess_data(filename, label, is_training=False,
                    resize_side_min=RESIZE_SIDE_MIN,
                    resize_side_max=RESIZE_SIDE_MAX,
                    output_height=CROP_HEIGHT,
                    output_width=CROP_WIDTH):
    '''
    Read image from file
    Preprocess image
    Return image and label
    '''
    image = _decode_jpeg(filename)

    if (is_training):
        return _preprocess_for_train(image, label, output_height,
                                     output_width, resize_side_min,
                                     resize_side_max)
    else:
        return _preprocess_for_val(image, label, output_height,
                                   output_width, resize_side_min)


def _decode_jpeg(image_bytes_tensor):
    '''
    Read JPEG encoded bytes from file and decode to 3D float Tensor
    '''
    # image_bytes_tensor = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_bytes_tensor, channels=3)
    return tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)


def _smallest_size_at_least(height, width, smallest_side):
    '''
    Computes new shape using smallest side so as to preserve original aspect
    ratio
    '''
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side=256.0):
    '''
    Resize image preserving the original aspect ratio
    '''
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width,
                                                    smallest_side)
    # Add batch dim of 1 to 4D shape [batch, height, width, channels]
    image = tf.expand_dims(image, 0)
    # Resize images to [batch, new_height, new_width, channels]
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    # Remove dims of size 1 from shape of tensor
    resized_image = tf.squeeze(resized_image)
    # Set shape inferring height and width
    resized_image.set_shape([None, None, 3])

    return resized_image


def _mean_image_subtraction(image, means):
    '''
    Subtract means from each image channel
    '''
    channels = tf.split(axis=2, num_or_size_splits=3, value=image)
    for i in range(3):
        channels[i] -= means[i]

    return tf.concat(axis=2, values=channels)


def _preprocess_for_train(image, label, output_height, output_width,
                          resize_side_min, resize_side_max):
    '''
    Resize image with scale sampled from range
    Take random crop to the scaled image
    Horizontally flip the image with probability 1/2
    Substract per colour means
    Don't normalize the data here as VGG was trained without normalization
    Return preprocessed image and label
    '''
    # Sample resizing scale from ['resize_size_min', 'resize_size_max']
    resize_side = tf.random_uniform([], minval=resize_side_min,
                                    maxval=resize_side_max + 1, dtype=tf.int32)
    image = _aspect_preserving_resize(image, resize_side)

    # Randomly crop resized image to target dims
    crop_image = tf.random_crop(image, [output_height, output_width, 3])

    # Randomly flip image horizontally
    flip_image = tf.image.random_flip_left_right(crop_image)

    # Subtract colour means
    preprocessed_image = _mean_image_subtraction(flip_image,
                                                 [R_MEAN, G_MEAN, B_MEAN])

    return preprocessed_image, label


def _preprocess_for_val(image, label, output_height,
                        output_width, resize_side):
    '''
    Take central crop to the scaled image
    Substract per color means
    Don't normalize the data here as VGG was trained without normalization
    Return preprocessed image and label
    '''
    image = _aspect_preserving_resize(image, resize_side)

    # Centrally crop resized image to target dims
    crop_image = tf.image.resize_image_with_crop_or_pad(image, output_height,
                                                        output_width)

    # Subtract colour means
    preprocessed_image = _mean_image_subtraction(crop_image,
                                                 [R_MEAN, G_MEAN, B_MEAN])

    return preprocessed_image, label
