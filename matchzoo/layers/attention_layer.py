"""An implementation of Attention Layer for Bimpm model."""

from keras import layers
from keras import backend as K
from keras.engine.topology import Layer

from matchzoo import utils


class AttentionLayer(Layer):
    """
    A keras implementation of Bimpm multi-perspective layer.

    For detailed information, see Bilateral Multi-Perspective
    Matching for Natural Language Sentences, section 3.2.
    """

    def __init__(
        self,
        att_dim: int,
        att_type: str = 'default',
        remove_diagnoal: bool = False,
        **kwargs
    ):
        """
        Class initialization.

        :param output_dim: dimensionality of output space.
        """
        self.att_type = att_type
        self.att_dim = att_dim
        self.remove_diagnoal = remove_diagnoal
        super(AttentionLayer, self).__init__(**kwargs)

    @property
    def get_att_type(cls):
        """Get the number of perspsectives that is True."""
        return cls.att_type

    def build(self, input_shapes):
        if not isinstance(input_shapes, list):
            raise ValueError('A attention layer should be called '
                             'on a list of inputs.')
        # input_shapes[0]: batch, time_steps, d
        hidden_dim = input_shapes[0][2]
        self.atten_W = self.add_weight(name='atten_W',
                                       shape=(hidden_dim,
                                              self.att_dim),
                                       initializer='uniform',
                                       trainable=True)
        self.diagnoal_W = self.add_weight(name='diagnoal_W',
                                          shape=(1,
                                                 1,
                                                 self.att_dim),
                                          initializer='uniform',
                                          trainable=True
                                          )
        super(AttentionLayer, self).build(input_shapes)

    def call(self, inputs: list):
        # calculate attention ==> a: [batch_size, len_1, len_2]

        if not isinstance(inputs, list):
            raise ValueError('A attention layer should be called '
                             'on a list of inputs.')

        # TODO(tjf) add mask
        lstm_lt, lstm_rt = inputs

        # [1, 1, d, 20]
        atten_W = K.expand_dims(self.atten_W, axis=0)
        atten_W = K.expand_dims(self.atten_W, axis=1)
        # [b, s, d, -1]
        lstm_lt = K.expand_dims(lstm_lt, axis=-1)
        atten_lt = K.sum(lstm_lt * atten_W, axis=2)
        lstm_rt = K.expand_dims(lstm_rt, axis=-1)
        atten_rt = K.expand_dims(lstm_rt * atten_W, axis=2)

        atten_lt = K.tanh(atten_lt) # [b, s, 20]
        atten_rt = K.tanh(atten_rt)

        atten_lt = atten_lt * self.diagnoal_W  # [b, s, 20]
        # atten_rt [b, s, 20]
        atten_value = K.batch_dot(atten_lt, atten_rt)  # [batch_size, s, s]

        # TODO(tjf) or remove: normalize
        # if self.remove_diagnoal:
        #     diagnoal = K.ones([len_1], tf.float32)  # [len1]
        #     diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
        #     diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
        #     atten_value = atten_value * diagnoal

        if len(inputs) == 4:
            mask_lt = inputs[2]
            mask_rt = inputs[3]
            atten_value = atten_value * K.expand_dims(mask_lt, axis=-1)
            atten_value = atten_value * K.expand_dims(mask_rt, axis=1)

        atten_value = K.softmax(atten_value)  # [batch_size, len_1, len_2]
        # if remove_diagnoal: atten_value = atten_value * diagnoal
        if len(inputs) == 4:
            mask_lt = inputs[2]
            mask_rt = inputs[3]
            atten_value = atten_value * K.expand_dims(mask_lt, axis=-1)
            atten_value = atten_value * K.expand_dims(mask_rt, axis=1)
        return atten_value

    def compute_output_shape(self, input_shapes):
        if not isinstance(input_shapes, list):
            raise ValueError('A attention layer should be called '
                             'on a list of inputs.')

        len_lt = input_shapes[0][1]
        len_rt = input_shapes[1][1]
        return (input_shapes[0][0], len_lt, len_rt)



attention_func = AttentionLayer(20)