"""An implementation of Attention Layer for Bimpm model."""

import tensorflow as tf
from keras import backend as K
from keras.engine import Layer


class AttentionLayer(Layer):
    """
    Layer that compute attention for BiMPM model.

    For detailed information, see Bilateral Multi-Perspective Matching for
    Natural Language Sentences, section 3.2.

    Reference:
    https://github.com/zhiguowang/BiMPM/blob/master/src/layer_utils.py#L145-L196

    Examples:
        >>> import matchzoo as mz
        >>> layer = mz.contrib.layers.AttentionLayer(att_dim=50)
        >>> layer.compute_output_shape([(32, 10, 100), (32, 40, 100)])
        (32, 10, 40)

    """

    def __init__(self,
                 att_dim: int,
                 att_type: str = 'default',
                 dropout_rate: float = 0.0):
        """
        class: `AttentionLayer` constructor.

        :param att_dim: int
        :param att_type: int
        """
        super(AttentionLayer, self).__init__()
        self._att_dim = att_dim
        self._att_type = att_type
        self._dropout_rate = dropout_rate

    @property
    def att_dim(self):
        """Get the attention dimension."""
        return self._att_dim

    @property
    def att_type(self):
        """Get the attention type."""
        return self._att_type

    def build(self, input_shapes):
        """
        Build the layer.

        :param input_shapes: input_shape_lt, input_shape_rt
        """
        if not isinstance(input_shapes, list):
            raise ValueError('A attention layer should be called '
                             'on a list of inputs.')

        hidden_dim_lt = input_shapes[0][2]
        hidden_dim_rt = input_shapes[1][2]

        self.attn_w1 = self.add_weight(name='attn_w1',
                                       shape=(hidden_dim_lt,
                                              self._att_dim),
                                       initializer='uniform',
                                       trainable=True)
        if hidden_dim_lt == hidden_dim_rt:
            self.attn_w2 = self.attn_w1
        else:
            self.attn_w2 = self.add_weight(name='attn_w2',
                                           shape=(hidden_dim_rt,
                                                  self._att_dim),
                                           initializer='uniform',
                                           trainable=True)
        # diagonal_W: (1, 1, a)
        self.diagonal_W = self.add_weight(name='diagonal_W',
                                          shape=(1,
                                                 1,
                                                 self._att_dim),
                                          initializer='uniform',
                                          trainable=True)
        self.built = True

    def call(self, x: list, **kwargs):
        """
        Calculate attention.

        :param x: [reps_lt, reps_rt]
        :return attn_prob: [b, s_lt, s_rt]
        """

        if not isinstance(x, list):
            raise ValueError('A attention layer should be called '
                             'on a list of inputs.')

        reps_lt, reps_rt = x

        attn_w1 = self.attn_w1
        attn_w1 = tf.expand_dims(tf.expand_dims(attn_w1, axis=0), axis=0)
        # => [1, 1, d, a]

        reps_lt = tf.expand_dims(reps_lt, axis=-1)
        attn_reps_lt = tf.reduce_sum(reps_lt * attn_w1, axis=2)
        # => [b, s_lt, d, -1]

        attn_w2 = self.attn_w2
        attn_w2 = tf.expand_dims(tf.expand_dims(attn_w2, axis=0), axis=0)
        # => [1, 1, d, a]

        reps_rt = tf.expand_dims(reps_rt, axis=-1)
        attn_reps_rt = tf.reduce_sum(reps_rt * attn_w2, axis=2)  # [b, s_rt, d, -1]

        attn_reps_lt = tf.tanh(attn_reps_lt)  # [b, s_lt, a]
        attn_reps_rt = tf.tanh(attn_reps_rt)  # [b, s_rt, a]

        # diagonal_W
        attn_reps_lt = attn_reps_lt * self.diagonal_W  # [b, s_lt, a]
        attn_reps_rt = tf.transpose(attn_reps_rt, (0, 2, 1))
        # => [b, a, s_rt]

        attn_value = K.batch_dot(attn_reps_lt, attn_reps_rt)  # [b, s_lt, s_rt]

        # Softmax operation
        attn_prob = tf.nn.softmax(attn_value)  # [b, s_lt, s_rt]

        # TODO(tjf) remove diagonal or not for normalization
        # if remove_diagonal: attn_value = attn_value * diagonal

        if len(x) == 4:
            mask_lt, mask_rt = x[2], x[3]
            attn_prob *= tf.expand_dims(mask_lt, axis=2)
            attn_prob *= tf.expand_dims(mask_rt, axis=1)

        return attn_prob

    def compute_output_shape(self, input_shapes):
        """Calculate the layer output shape."""
        if not isinstance(input_shapes, list):
            raise ValueError('A attention layer should be called '
                             'on a list of inputs.')
        input_shape_lt, input_shape_rt = input_shapes[0], input_shapes[1]
        return input_shape_lt[0], input_shape_lt[1], input_shape_rt[1]
