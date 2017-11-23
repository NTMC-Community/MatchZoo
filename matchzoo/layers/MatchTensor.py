from __future__ import absolute_import

import numpy as np

import keras
from keras import backend as K
from keras.engine import Layer
from keras.engine import InputSpec

class MatchTensor(Layer):
    """Layer that computes a matching matrix between samples in two tensors.
    # Arguments
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    """

    def __init__(self, channel, normalize=False, init_diag=False, **kwargs):
        super(MatchTensor, self).__init__(**kwargs)
        self.channel = channel
        self.normalize = normalize
        self.init_diag = init_diag
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `MatchTensor` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1[0] != shape2[0]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (shape1[0], shape2[0]) +
                'Layer shapes: %s, %s' % (shape1, shape2))
        if self.init_diag:
            if shape1[2] != shape2[2]:
                raise ValueError( 'Use init_diag need same embedding shape.' )
            M_diag = np.float32(np.random.uniform(-0.05, 0.05, [self.channel, shape1[2], shape2[2]]))
            for i in range(self.channel):
                for j in range(shape1[2]):
                    M_diag[i][j][j] = 1.0
            self.M = self.add_weight( name='M', 
                                   shape=(self.channel, shape1[2], shape2[2]),
                                   initializer=M_diag,
                                   trainable=True )
        else:
            self.M = self.add_weight( name='M', 
                                   shape=(self.channel, shape1[2], shape2[2]),
                                   initializer='uniform',
                                   trainable=True )

    def call(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        if self.normalize:
            x1 = K.l2_normalize(x1, axis=2)
            x2 = K.l2_normalize(x2, axis=2)
        output = K.tf.einsum('abd,fde,ace->afbc', x1, self.M, x2)
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `MatchTensor` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if len(shape1) != 3 or len(shape2) != 3:
            raise ValueError('A `MatchTensor` layer should be called '
                             'on 2 inputs with 3 dimensions.')
        if shape1[0] != shape2[0]:
            raise ValueError('A `MatchTensor` layer should be called '
                             'on 2 inputs with same 0,2 dimensions.')

        output_shape = [shape1[0], self.channel, shape1[1], shape2[1]]

        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'channel': self.channel,
            'normalize': self.normalize,
            'init_diag': self.init_diag,
        }
        base_config = super(MatchTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def match(inputs, axes, normalize=False, **kwargs):
    """Functional interface to the `MatchTensor` layer.
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
    return MatchTensor(normalize=normalize, **kwargs)(inputs)
