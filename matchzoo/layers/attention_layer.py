"""An implementation of Attention Layer for Bimpm model."""

from keras import backend as K
from keras.engine import Layer


class AttentionLayer(Layer):
    """
    Layer that compute attention for Bimpm model.

    For detailed information, see Bilateral Multi-Perspective Matching for
    Natural Language Sentences, section 3.2.

    Reference:
    https://github.com/zhiguowang/BiMPM/blob/master/src/layer_utils.py#L145-L196

    Examples:
        >>> import matchzoo as mz
        >>> layer = mz.layers.AttentionLayer(att_dim=50)

    """

    def __init__(self,
                 att_dim: int,
                 att_type: str = 'default',
                 remove_diagnoal: bool = False,
                 **kwargs):
        """
        class: `AttentionLayer` constructor.

        :param att_dim: int
        :param att_type: int
        :param remove_diagnoal: bool
        """
        self.att_dim = att_dim
        self.att_type = att_type
        self.remove_diagnoal = remove_diagnoal
        super(AttentionLayer, self).__init__(**kwargs)

    @property
    def get_att_type(cls):
        """Get the number of perspsectives that is True."""
        return cls.att_type

    def build(self, input_shapes):
        """
        Build the layer.

        :param input_shapes: input_shapes
        """
        if not isinstance(input_shapes, list):
            raise ValueError('A attention layer should be called '
                             'on a list of inputs.')
        # input_shapes[0]: batch, time_steps, d
        hidden_dim = input_shapes[0][2]
        self.attn_kernel = self.add_weight(name='attn_kernel',
                                           shape=(hidden_dim,
                                                  self.att_dim),
                                           initializer='uniform',
                                           trainable=True)
        self.diagnoal_W = self.add_weight(name='diagnoal_W',
                                          shape=(1,
                                                 1,
                                                 self.att_dim),
                                          initializer='uniform',
                                          trainable=True)
        super(AttentionLayer, self).build(input_shapes)

    def call(self, x: list, **kwargs):
        """
        Calculate attention.

        :param x: [reps_lt, reps_rt]
        :return attn_prob: [batch_size, len_1, len_2]
        """

        if not isinstance(x, list):
            raise ValueError('A attention layer should be called '
                             'on a list of inputs.')

        reps_lt, reps_rt = x

        # [1, 1, d, 20]
        attn_kernel = self.attn_kernel

        attn_kernel = K.expand_dims(attn_kernel, axis=0)
        attn_kernel = K.expand_dims(attn_kernel, axis=1)
        # [b, s, d, -1]
        reps_lt = K.expand_dims(reps_lt, axis=-1)
        attn_reps_lt = K.sum(reps_lt * attn_kernel, axis=2)
        reps_rt = K.expand_dims(reps_rt, axis=-1)
        attn_reps_rt = K.sum(reps_rt * attn_kernel, axis=2)

        # Tanh
        attn_reps_lt = K.tanh(attn_reps_lt)  # [b, s, 20]
        attn_reps_rt = K.tanh(attn_reps_rt)

        # diagnoal_W
        attn_reps_lt = attn_reps_lt * self.diagnoal_W  # [b, s, 20]
        attn_reps_rt = K.permute_dimensions(attn_reps_rt, (0, 2, 1))

        # [batch_size, s, s]
        attn_value = K.batch_dot(attn_reps_lt, attn_reps_rt)

        # TODO(tjf) or remove: normalize
        # if self.remove_diagnoal:
        #     diagnoal = K.ones([len_1], tf.float32)  # [len1]
        #     diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
        #     diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
        #     atten_value = atten_value * diagnoal

        if len(x) == 4:
            mask_lt = x[2]
            mask_rt = x[3]
            attn_value = attn_value * K.expand_dims(mask_lt, axis=2)
            attn_value = attn_value * K.expand_dims(mask_rt, axis=1)

        # softmax
        attn_prob = K.softmax(attn_value)  # [batch_size, len_1, len_2]

        # if remove_diagnoal: attn_value = attn_value * diagnoal
        if len(x) == 4:
            mask_lt = x[2]
            mask_rt = x[3]
            attn_prob = attn_prob * K.expand_dims(mask_lt, axis=2)
            attn_prob = attn_prob * K.expand_dims(mask_rt, axis=1)

        return attn_prob

    def compute_output_shape(self, input_shapes):
        """Calculate the layer output shape."""
        if not isinstance(input_shapes, list):
            raise ValueError('A attention layer should be called '
                             'on a list of inputs.')
        return input_shapes[0][0], input_shapes[0][1], input_shapes[1][1]
