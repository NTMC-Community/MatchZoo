"""An implementation of Matching Tensor Layer."""
import typing

import numpy as np
from keras import backend as K
from keras.engine import Layer
from keras.initializers import constant


class MatchingTensorLayer(Layer):
    """
    Layer that captures the basic interactions between two tensors.

    :param channels: Number of word interaction tensor channels
    :param normalize: Whether to L2-normalize samples along the
        dot product axis before taking the dot product.
        If set to True, then the output of the dot product
        is the cosine proximity between the two samples.
    :param init_diag: Whether to initialize the diagonal elements
        of the matrix.
    :param kwargs: Standard layer keyword arguments.

    Examples:
        >>> import matchzoo as mz
        >>> layer = mz.contrib.layers.MatchingTensorLayer(channels=4,
        ...                                               normalize=True,
        ...                                               init_diag=True)
        >>> num_batch, left_len, right_len, num_dim = 5, 3, 2, 10
        >>> layer.build([[num_batch, left_len, num_dim],
        ...              [num_batch, right_len, num_dim]])

    """

    def __init__(self, channels: int = 4, normalize: bool = True,
                 init_diag: bool = True, **kwargs):
        """:class:`MatchingTensorLayer` constructor."""
        super().__init__(**kwargs)
        self._channels = channels
        self._normalize = normalize
        self._init_diag = init_diag
        self._shape1 = None
        self._shape2 = None

    def build(self, input_shape: list):
        """
        Build the layer.

        :param input_shape: the shapes of the input tensors,
            for MatchingTensorLayer we need two input tensors.
        """
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `MatchingTensorLayer` layer should be called '
                             'on a list of 2 inputs.')
        self._shape1 = input_shape[0]
        self._shape2 = input_shape[1]
        for idx in (0, 2):
            if self._shape1[idx] != self._shape2[idx]:
                raise ValueError(
                    'Incompatible dimensions: '
                    f'{self._shape1[idx]} != {self._shape2[idx]}.'
                    f'Layer shapes: {self._shape1}, {self._shape2}.'
                )

        if self._init_diag:
            interaction_matrix = np.float32(
                np.random.uniform(
                    -0.05, 0.05,
                    [self._channels, self._shape1[2], self._shape2[2]]
                )
            )
            for channel_index in range(self._channels):
                np.fill_diagonal(interaction_matrix[channel_index], 0.1)
            self.interaction_matrix = self.add_weight(
                name='interaction_matrix',
                shape=(self._channels, self._shape1[2], self._shape2[2]),
                initializer=constant(interaction_matrix),
                trainable=True
            )
        else:
            self.interaction_matrix = self.add_weight(
                name='interaction_matrix',
                shape=(self._channels, self._shape1[2], self._shape2[2]),
                initializer='uniform',
                trainable=True
            )
        super(MatchingTensorLayer, self).build(input_shape)

    def call(self, inputs: list, **kwargs) -> typing.Any:
        """
        The computation logic of MatchingTensorLayer.

        :param inputs: two input tensors.
        """
        x1 = inputs[0]
        x2 = inputs[1]
        # Normalize x1 and x2
        if self._normalize:
            x1 = K.l2_normalize(x1, axis=2)
            x2 = K.l2_normalize(x2, axis=2)

        # b = batch size
        # l = length of `x1`
        # r = length of `x2`
        # d, e = embedding size
        # c = number of channels
        # output = [b, c, l, r]
        output = K.tf.einsum(
            'bld,cde,bre->bclr',
            x1, self.interaction_matrix, x2
        )
        return output

    def compute_output_shape(self, input_shape: list) -> tuple:
        """
        Calculate the layer output shape.

        :param input_shape: the shapes of the input tensors,
            for MatchingTensorLayer we need two input tensors.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `MatchingTensorLayer` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if len(shape1) != 3 or len(shape2) != 3:
            raise ValueError('A `MatchingTensorLayer` layer should be called '
                             'on 2 inputs with 3 dimensions.')
        if shape1[0] != shape2[0] or shape1[2] != shape2[2]:
            raise ValueError('A `MatchingTensorLayer` layer should be called '
                             'on 2 inputs with same 0,2 dimensions.')

        output_shape = [shape1[0], self._channels, shape1[1], shape2[1]]
        return tuple(output_shape)
