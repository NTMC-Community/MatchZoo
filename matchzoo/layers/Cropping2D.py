# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np

import keras
from keras import backend as K
from keras.engine import Layer
from keras.engine import InputSpec

class Cropping2D(Layer):
    def __init__(self, cropping=((0,0), (0,0)), dim_ordering='default', **kwargs):
        super(Cropping2D, self).__init__(**kwargs)
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.cropping = tuple(cropping)
        assert len(self.cropping) == 2, 'cropping must be a tuple length of 2'
        assert len(self.cropping[0]) == 2, 'cropping[0] must be a tuple length of 2'
        assert len(self.cropping[1]) == 2, 'cropping[1] must be a tuple length of 2'
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)]
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return (input_shape[0],
                    input_shape[1],
                    input_shape[2] - self.cropping[0][0] - self.cropping[0][1],
                    input_shape[3] - self.cropping[1][0] - self.cropping[1][1])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    input_shape[1] - self.cropping[0][0] - self.cropping[0][1],
                    input_shape[2] - self.cropping[1][0] - self.cropping[1][1],
                    input_shape[3])
        else:
            return Exception('Invalid dim_ordering: ' + self.dim_ordering)
    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        if self.dim_ordering == 'th':
            return x[:,
                    :,
                    self.cropping[0][0]:input_shape[2]-self.cropping[0][1],
                    self.cropping[1][0]:input_shape[3]-self.cropping[1][1]]
        elif self.dim_ordering == 'tf':
            return x[:,
                    self.cropping[0][0]:input_shape[1]-self.cropping[0][1],
                    self.cropping[1][0]:input_shape[2]-self.cropping[1][1],
                    :]
    def get_config(self):
        config={'cropping': self.cropping}
        base_config = super(Cropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
