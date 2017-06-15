from __future__ import absolute_import

import numpy as np

import keras
from keras import backend as K
from keras.engine import Layer
from keras.engine import InputSpec

class Match(Layer):
    """Layer that computes a matching matrix between samples in two tensors.
    # Arguments
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    """

    def __init__(self, normalize=False, **kwargs):
        super(Match, self).__init__(**kwargs)
        self.normalize = normalize
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Match` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1[0] != shape2[0]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (shape1[0], shape2[0]) +
                'Layer shapes: %s, %s' % (shape1, shape2))
        if shape1[2] != shape2[2]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (shape1[2], shape2[2]) +
                'Layer shapes: %s, %s' % (shape1, shape2))

    def call(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        if self.normalize:
            x1 = K.l2_normalize(x1, axis=2)
            x2 = K.l2_normalize(x2, axis=2)
        output = K.tf.einsum('abd,acd->abc', x1, x2)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Match` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if len(shape1) != 3 or len(shape2) != 3:
            raise ValueError('A `Match` layer should be called '
                             'on 2 inputs with 3 dimensions.')
        if shape1[0] != shape2[0] or shape1[2] != shape2[2]:
            raise ValueError('A `Match` layer should be called '
                             'on 2 inputs with same 0,2 dimensions.')

        output_shape = [shape1[0], shape1[1], shape2[1]]

        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'normalize': self.normalize,
        }
        base_config = super(Match, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def match(inputs, axes, normalize=False, **kwargs):
    """Functional interface to the `Match` layer.
    # Arguments
        inputs: A list of input tensors (with exact 2 tensors).
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the dot product matching matrix of the samples 
        from the inputs.
    """
    return Match(normalize=normalize, **kwargs)(inputs)
