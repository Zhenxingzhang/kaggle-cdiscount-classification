import numpy as np
import tensorflow as tf

from src.common import consts


def _weight_variable(shape):
    return tf.get_variable(name='weights',
                           initializer=tf.truncated_normal(shape, stddev=0.1))


def _bias_variable(shape):
    return tf.get_variable(name='biases',
                           initializer=tf.constant(0.1, shape=shape))


def _variable_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar(name + '/mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name + '/stddev', stddev)


def _activation_summaries(x, name):
    tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))
    tf.summary.histogram(name, x)
    _variable_summaries(x, name)


def _fc_layer(layer_name, input_tensor, no_filters, act=tf.nn.relu,
              dropout=None, summaries=False):
    '''
    Flatten input_tensor if needed, non-linearize with ReLU and optionally
    apply dropout
    '''
    with tf.variable_scope(layer_name):
        # Reshape input tensor to flatten tensor if needed
        input_shape = input_tensor.get_shape()
        if len(input_shape) == 4:
            input_dim = np.int(np.product(input_shape[1:]))
            input_tensor = tf.reshape(input_tensor, [-1, input_dim])
        elif len(input_shape) == 2:
            input_dim = input_shape[-1].value
        else:
            raise RuntimeError('Input Tensor Shape: {}'.format(input_shape))

        weight = _weight_variable([input_dim, no_filters])
        bias = _bias_variable([no_filters])
        preactivate = tf.nn.bias_add(tf.matmul(input_tensor, weight), bias)
        activation = act(preactivate, "activation")

        if dropout is not None:
            activation = tf.nn.dropout(activation, dropout)

        if summaries:
            _variable_summaries(weight, '/weights')
            _variable_summaries(bias, '/biases')
            _activation_summaries(activation, '/activations')

        return activation


def dense_neural_network(layers, gamma=0.1):
    n_x = layers[0]
    n_y = layers[-1]
    L = len(layers)
    summaries = []
    # Ws = []

    with tf.name_scope("placeholders"):
        # x = tf.placeholder(dtype=tf.float32, shape=(n_x, None), name="x")
        x = tf.placeholder(dtype=tf.float32, shape=(None, n_x), name="x")
        y = tf.placeholder(dtype=tf.int32, shape=(None), name="y")

    with tf.name_scope("hidden_layers"):
        a = x
        for l in range(1, len(layers) - 1):
            # W = tf.Variable(np.random.randn(layers[l], layers[l - 1]) / tf.sqrt(layers[l - 1] * 1.0),
            # dtype=tf.float32, name="W" + str(l))
            W_ = tf.Variable(tf.random_normal([layers[l - 1], layers[l]], stddev=0.35), name="W" + str(l))

            # Ws.append(W_)
            summaries.append(tf.summary.histogram('W' + str(l), W_))
            # b = tf.Variable(np.zeros((layers[l], 1)), dtype=tf.float32, name="b" + str(l))
            b = tf.Variable(tf.zeros([layers[l]]), dtype=tf.float32, name="b" + str(l))
            summaries.append(tf.summary.histogram('b' + str(l), b))
            # z = tf.matmul(W, a) + b
            z = tf.matmul(a, W_) + b
            a = tf.nn.relu(z)

    # W = tf.Variable(np.random.randn(layers[L - 1], layers[L - 2]) / tf.sqrt(layers[L - 2] * 1.0), dtype=tf.float32,
    #                 name="W" + str(L - 1))

    W = tf.Variable(tf.random_normal([layers[L - 2], layers[L-1]], stddev=0.35), name="W" + str(L-1))
    summaries.append(tf.summary.histogram('W' + str(L - 1), W))

    b = tf.Variable(np.zeros([layers[L - 1]]), dtype=tf.float32, name="b" + str(L - 1))
    summaries.append(tf.summary.histogram('b' + str(L - 1), b))
    z = tf.matmul(a, W) + b

    with tf.name_scope("cost"):
        cost = gamma * tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z))  # +\
        # gamma * tf.reduce_sum([tf.nn.l2_loss(w) for w in Ws])
        summaries.append(tf.summary.scalar('cost', cost))

    output = tf.nn.softmax(z, name=consts.OUTPUT_NODE_NAME)

    return cost, output, x, y, summaries
