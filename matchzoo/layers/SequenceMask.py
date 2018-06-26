# coding: utf8
from __future__ import print_function
from __future__ import absolute_import

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class SequenceMask(Layer):

    def __init__(self, text_maxlen, **kwargs):
        super(SequenceMask, self).__init__(**kwargs)
        self.text_maxlen = text_maxlen

    def build(self, input_shape):
        super(SequenceMask, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, input_len):
        input_len = K.tf.squeeze(input_len, axis=-1)  # [batch, 1] -> [batch, ]
        mask = K.tf.sequence_mask(input_len, self.text_maxlen, dtype=K.tf.float32)
        return mask

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.text_maxlen)

    def get_config(self):
        config = {
            'text_maxlen': self.text_maxlen,
        }
        base_config = super(SequenceMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
