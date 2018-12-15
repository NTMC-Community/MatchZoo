"""An implementation of Matching Layer."""
import typing

from keras import backend as K
from keras.engine import Layer


class MatchingLayer(Layer):
    """
    Layer that computes a matching matrix between samples in two tensors.

    Examples:
        >>> import matchzoo as mz
        >>> layer = mz.layers.MatchingLayer()

    """

    def __init__(self, normalize: bool = False,
                 matching_type: str = 'dot', **kwargs):
        """
        :class:`MatchingLayer` constructor.

        :param normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        :param matching_type: the similarity function for matching
        :param **kwargs: Standard layer keyword arguments.
        """
        super(MatchingLayer, self).__init__(**kwargs)
        self.normalize = normalize
        self.matching_type = matching_type
        self.supports_masking = True
        self.valid_matching_type = ['dot', 'mul', 'plus', 'minus', 'concat']
        if matching_type not in self.valid_matching_type:
            raise ValueError("{} is not a valid matching type,"
                             " {} expected.".format(self.matching_type,
                                                    self.valid_matching_type))

    def build(self, input_shape: list):
        """
        Build the layer.

        :param input_shape: the shapes of the input tensors,
            for MatchingLayer we need tow input tensors.
        """
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `MatchingLayer` layer should be called '
                             'on a list of 2 inputs.')
        self.shape1 = input_shape[0]
        self.shape2 = input_shape[1]
        if self.shape1[0] != self.shape2[0]:
            raise ValueError(
                'Incompatible dimensions '
                '{} != {}. Layer shapes: {}, {}.'.format(self.shape1[0],
                                                         self.shape2[0],
                                                         self.shape1,
                                                         self.shape2))
        if self.shape1[2] != self.shape2[2]:
            raise ValueError(
                'Incompatible dimensions '
                '{} != {}. Layer shapes: {}, {}.'.format(self.shape1[2],
                                                         self.shape2[2],
                                                         self.shape1,
                                                         self.shape2))

    def call(self, inputs: list) -> typing.Any:
        """
        The computation logic of MatchingLayer.

        :param inputs: two input tensors.
        """
        x1 = inputs[0]
        x2 = inputs[1]
        if self.matching_type in ['dot']:
            if self.normalize:
                x1 = K.l2_normalize(x1, axis=2)
                x2 = K.l2_normalize(x2, axis=2)
            output = K.tf.einsum('abd,acd->abc', x1, x2)
            output = K.tf.expand_dims(output, 3)
        elif self.matching_type in ['mul', 'plus', 'minus']:
            x1_exp = K.tf.stack([x1] * self.shape2[1], 2)
            x2_exp = K.tf.stack([x2] * self.shape1[1], 1)
            if self.matching_type == 'mul':
                output = x1_exp * x2_exp
            elif self.matching_type == 'plus':
                output = x1_exp + x2_exp
            elif self.matching_type == 'minus':
                output = x1_exp - x2_exp
        elif self.matching_type in ['concat']:
            x1_exp = K.tf.stack([x1] * self.shape2[1], axis=2)
            x2_exp = K.tf.stack([x2] * self.shape1[1], axis=1)
            output = K.tf.concat([x1_exp, x2_exp], axis=3)

        return output

    def compute_output_shape(self, input_shape: list) -> tuple:
        """
        Calculate the layer output shape.

        :param input_shape: the shapes of the input tensors,
            for MatchingLayer we need tow input tensors.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `MatchingLayer` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if len(shape1) != 3 or len(shape2) != 3:
            raise ValueError('A `MatchingLayer` layer should be called '
                             'on 2 inputs with 3 dimensions.')
        if shape1[0] != shape2[0] or shape1[2] != shape2[2]:
            raise ValueError('A `MatchingLayer` layer should be called '
                             'on 2 inputs with same 0,2 dimensions.')

        if self.matching_type in ['dot']:
            output_shape = [shape1[0], shape1[1], shape2[1], 1]
        elif self.matching_type in ['mul', 'plus', 'minus']:
            output_shape = [shape1[0], shape1[1], shape2[1], shape1[2]]
        elif self.matching_type in ['concat']:
            output_shape = [shape1[0], shape1[1], shape2[1],
                            shape1[2] + shape2[2]]

        return tuple(output_shape)

    def compute_mask(self, inputs: list, mask: list = None) -> typing.Any:
        """Compute input mask. Undefine in this layer."""
        return None

    def get_config(self) -> dict:
        """Get the config dict of MatchingLayer."""
        config = {
            'normalize': self.normalize,
            'matching_type': self.matching_type,
        }
        base_config = super(MatchingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
