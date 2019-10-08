"""An implementation of EncodingModule for DIIN model."""

import tensorflow as tf
from keras import backend as K
from keras.engine import Layer

from matchzoo.contrib.layers import DecayingDropoutLayer


class EncodingLayer(Layer):
    """
    Apply a self-attention layer and a semantic composite fuse gate
    to compute the encoding result of one tensor.

    :param initial_keep_rate: the initial_keep_rate parameter of
        DecayingDropoutLayer.
    :param decay_interval: the decay_interval parameter of
        DecayingDropoutLayer.
    :param decay_rate: the decay_rate parameter of DecayingDropoutLayer.
    :param kwargs: standard layer keyword arguments.

    Example:
        >>> import matchzoo as mz
        >>> layer = mz.contrib.layers.EncodingLayer(1.0, 10000, 0.977)
        >>> num_batch, left_len, num_dim = 5, 32, 10
        >>> layer.build([num_batch, left_len, num_dim])
    """

    def __init__(self,
                 initial_keep_rate: float,
                 decay_interval: int,
                 decay_rate: float,
                 **kwargs):
        """:class: 'EncodingLayer' constructor."""
        super(EncodingLayer, self).__init__(**kwargs)
        self._initial_keep_rate = initial_keep_rate
        self._decay_interval = decay_interval
        self._decay_rate = decay_rate
        self._w_itr_att = None
        self._w1 = None
        self._w2 = None
        self._w3 = None
        self._b1 = None
        self._b2 = None
        self._b3 = None

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: the shape of the input tensor,
            for EncodingLayer we need one input tensor.
        """
        d = input_shape[-1]

        self._w_itr_att = self.add_weight(
            name='w_itr_att', shape=(3 * d,), initializer='glorot_uniform')
        self._w1 = self.add_weight(
            name='w1', shape=(2 * d, d,), initializer='glorot_uniform')
        self._w2 = self.add_weight(
            name='w2', shape=(2 * d, d,), initializer='glorot_uniform')
        self._w3 = self.add_weight(
            name='w3', shape=(2 * d, d,), initializer='glorot_uniform')
        self._b1 = self.add_weight(
            name='b1', shape=(d,), initializer='zeros')
        self._b2 = self.add_weight(
            name='b2', shape=(d,), initializer='zeros')
        self._b3 = self.add_weight(
            name='b3', shape=(d,), initializer='zeros')

        super(EncodingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        The computation logic of EncodingLayer.

        :param inputs: an input tensor.
        """
        # Scalar dimensions referenced here:
        #   b = batch size
        #   p = inputs.shape()[1]
        #   d = inputs.shape()[2]

        # The input shape is [b, p, d]
        # shape = [b, 1, p, d]
        x = tf.expand_dims(inputs, 1) * 0
        # shape = [b, 1, d, p]
        x = tf.transpose(x, (0, 1, 3, 2))
        # shape = [b, p, d, p]
        mid = x + tf.expand_dims(inputs, -1)
        # shape = [b, p, d, p]
        up = tf.transpose(mid, (0, 3, 2, 1))
        # shape = [b, p, 3d, p]
        inputs_concat = tf.concat([up, mid, up * mid], axis=2)

        # Self-attention layer.
        # shape = [b, p, p]
        A = K.dot(self._w_itr_att, inputs_concat)
        # shape = [b, p, p]
        SA = tf.nn.softmax(A, axis=2)
        # shape = [b, p, d]
        itr_attn = K.batch_dot(SA, inputs)

        # Semantic composite fuse gate.
        # shape = [b, p, 2d]
        inputs_attn_concat = tf.concat([inputs, itr_attn], axis=2)
        concat_dropout = DecayingDropoutLayer(
            initial_keep_rate=self._initial_keep_rate,
            decay_interval=self._decay_interval,
            decay_rate=self._decay_rate
        )(inputs_attn_concat)
        # shape = [b, p, d]
        z = tf.tanh(K.dot(concat_dropout, self._w1) + self._b1)
        # shape = [b, p, d]
        r = tf.sigmoid(K.dot(concat_dropout, self._w2) + self._b2)
        # shape = [b, p, d]
        f = tf.sigmoid(K.dot(concat_dropout, self._w3) + self._b3)
        # shape = [b, p, d]
        encoding = r * inputs + f * z

        return encoding
