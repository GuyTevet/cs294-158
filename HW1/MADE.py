"""

This MADE implementation is taken from
https://github.com/ikrets/CS294-158-homeworks/blob/master/HW1_3_25epochs.ipynb

"""
import tensorflow as tf
import numpy as np

class MadeInput:
    def __init__(self, scope, inp, depth):
        with tf.variable_scope(scope):
            self.width = inp.shape[1].value
            self.height = inp.shape[2].value
            self.D = inp.shape[3].value
            self.depth = depth

            self.units = tf.reshape(
                tf.one_hot(inp, depth=self.depth, dtype=tf.float32),
                shape=(tf.shape(inp)[0],
                       self.width,
                       self.height,
                       self.depth * self.D))
            self.m = np.arange(self.D)
            self.m = np.repeat(self.m, self.depth, axis=-1)


class MadeHiddenWithAuxiliary:
    def __init__(self, scope, made_prev_layer, auxiliary, unit_count):
        with tf.variable_scope(scope):
            self.width = made_prev_layer.width
            self.height = made_prev_layer.height
            self.D = made_prev_layer.D
            self.depth = made_prev_layer.depth

            self.m = np.mod(np.arange(unit_count), self.D - 1)

            self.units = tf.concat([made_prev_layer.units, auxiliary], axis=-1)
            ext_input_length = made_prev_layer.m.shape[-1] + auxiliary.shape[-1].value

            self.weight_mask = np.ones((unit_count,
                                        ext_input_length),
                                       dtype=np.bool)
            self.weight_mask[:self.m.shape[-1],
            :made_prev_layer.m.shape[-1]] = self.m[:, np.newaxis] >= made_prev_layer.m[np.newaxis, :]
            self.W = tf.get_variable("W", shape=(self.width,
                                                 self.height,
                                                 unit_count,
                                                 ext_input_length))
            self.b = tf.get_variable("b",
                                     shape=(self.width,
                                            self.height,
                                            unit_count))

            self.units = tf.einsum('whij,bwhj->bwhi', self.W * self.weight_mask, self.units) + self.b
            self.units = tf.nn.relu(self.units)


class MadeHidden:
    def __init__(self, scope, made_prev_layer, unit_count):
        with tf.variable_scope(scope):
            self.width = made_prev_layer.width
            self.height = made_prev_layer.height
            self.D = made_prev_layer.D
            self.depth = made_prev_layer.depth

            self.m = np.mod(np.arange(unit_count), self.D - 1)

            self.weight_mask = self.m[:, np.newaxis] >= made_prev_layer.m[np.newaxis, :]
            self.W = tf.get_variable("W", shape=(self.width,
                                                 self.height,
                                                 unit_count,
                                                 made_prev_layer.m.shape[-1]))
            self.b = tf.get_variable("b", shape=(self.width,
                                                 self.height,
                                                 unit_count))

            self.units = tf.einsum('whij,bwhj->bwhi', self.W * self.weight_mask,
                                   made_prev_layer.units) + self.b
            self.units = tf.nn.relu(self.units)


class MadeOutput:
    def __init__(self, scope, made_input_layer, made_prev_layer, auxiliary):
        with tf.variable_scope(scope):
            self.width = made_prev_layer.width
            self.height = made_prev_layer.height
            self.D = made_prev_layer.D
            self.depth = made_prev_layer.depth

            self.m = np.repeat(np.arange(self.D), self.depth)
            self.weight_mask = self.m[:, np.newaxis] > made_prev_layer.m[np.newaxis, :]
            self.W = tf.get_variable("W", shape=(self.width,
                                                 self.height,
                                                 self.D * self.depth,
                                                 made_prev_layer.m.shape[-1]))
            self.b = tf.get_variable("b", shape=(self.width,
                                                 self.height,
                                                 self.D * self.depth))

            self.direct_mask = np.repeat(np.tril(np.ones(self.D), -1), self.depth).reshape((self.D, -1))
            self.direct_mask = np.repeat(self.direct_mask, self.depth, axis=0)
            self.A = tf.get_variable("A", shape=(self.width,
                                                 self.height,
                                                 self.D * self.depth,
                                                 self.D * self.depth))

            self.units = tf.einsum('whij,bwhj->bwhi',
                                   self.W * self.weight_mask,
                                   made_prev_layer.units) + self.b
            self.units += tf.einsum('whij,bwhj->bwhi',
                                    self.A * self.direct_mask,
                                    made_input_layer.units)

            self.unconnected_W = tf.get_variable('unconnected_W',
                                                 shape=(self.width,
                                                        self.height,
                                                        self.depth,
                                                        auxiliary.shape[-1].value))
            self.unconnected_b = tf.get_variable('unconnected_b',
                                                 shape=(self.width,
                                                        self.height,
                                                        self.depth))
            self.unconnected_out = tf.einsum('whij,bwhj->bwhi',
                                             self.unconnected_W,
                                             auxiliary) + self.unconnected_b
            self.units = tf.concat([self.unconnected_out,
                                    self.units[:, :, :, self.depth:]],
                                   axis=3)
            self.units = tf.reshape(self.units, shape=(tf.shape(self.units)[0],
                                                       self.width,
                                                       self.height,
                                                       self.D,
                                                       self.depth))