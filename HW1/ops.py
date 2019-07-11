import tensorflow as tf
import numpy as np

def mask(size, type_A):
    m = np.zeros((size, size), dtype=np.float32)
    m[:size // 2, :] = 1
    m[size // 2, :size // 2] = 1
    if not type_A:
        m[size // 2, size // 2] = 1
    return m

def conv_masked(inp, name, size, in_channels=128, out_channels=128, type_A=False):
    conv_filter = tf.get_variable(name=name + '_filter',
                                  shape=(size, size, in_channels, out_channels),
                                  trainable=True)
    conv_bias = tf.get_variable(name=name + '_bias',
                                shape=(inp.shape[1].value, inp.shape[2].value, out_channels),
                                trainable=True)

    masked_conv_filter = conv_filter * mask(size, type_A)[:, :, np.newaxis, np.newaxis]
    return tf.nn.conv2d(inp, masked_conv_filter, strides=[1, 1, 1, 1], padding='SAME',
                        name=name) + conv_bias


def conv_1x1(inp, name, in_channels, out_channels):
    conv_filter = tf.get_variable(name=name + '_filter',
                                  shape=(1, 1, in_channels, out_channels),
                                  trainable=True)
    conv_bias = tf.get_variable(name=name + '_bias',
                                shape=(inp.shape[1].value, inp.shape[2].value, out_channels),
                                trainable=True)
    return tf.nn.conv2d(inp, conv_filter, strides=[1, 1, 1, 1], padding='SAME', name=name) + conv_bias


def res_block(inp, scope, channels=128):
    with tf.variable_scope(scope):
        res = tf.nn.relu(inp)
        res = conv_1x1(res, 'conv1x1_downsample', channels, channels // 2)
        res = tf.nn.relu(res)
        res = conv_masked(res, 'conv3x3', 3, channels // 2, channels // 2)
        res = tf.nn.relu(res)
        res = conv_1x1(res, 'conv1x1_upsample', channels // 2, channels)

        return inp + res