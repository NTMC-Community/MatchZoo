"""An implementation of Dynamic Pooling Layer."""
import typing

import tensorflow as tf
from keras.engine import Layer


class DynamicPoolingLayer(Layer):
    """
    Layer that computes dynamic pooling of one tensor.

    :param psize1: pooling size of dimension 1
    :param psize2: pooling size of dimension 2
    :param kwargs: Standard layer keyword arguments.

    Examples:
        >>> import matchzoo as mz
        >>> layer = mz.layers.DynamicPoolingLayer(3, 2)
        >>> num_batch, left_len, right_len, num_dim = 5, 3, 2, 10
        >>> layer.build([[num_batch, left_len, right_len, num_dim],
        ...              [num_batch, left_len, right_len, 3]])

    """

    def __init__(self,
                 psize1: int,
                 psize2: int,
                 **kwargs):
        """:class:`DynamicPoolingLayer` constructor."""
        super().__init__(**kwargs)
        self._psize1 = psize1
        self._psize2 = psize2

    def build(self, input_shape: typing.List[int]):
        """
        Build the layer.

        :param input_shape: the shapes of the input tensors,
            for DynamicPoolingLayer we need tow input tensors.
        """
        super().build(input_shape)
        input_shape_one = input_shape[0]
        self._msize1 = input_shape_one[1]
        self._msize2 = input_shape_one[2]

    def call(self, inputs: list, **kwargs) -> typing.Any:
        """
        The computation logic of DynamicPoolingLayer.

        :param inputs: two input tensors.
        """
        self._validate_dpool_size()
        x, dpool_index = inputs
        dpool_shape = tf.shape(dpool_index)
        batch_index_one = tf.expand_dims(
            tf.expand_dims(
                tf.range(dpool_shape[0]), axis=-1),
            axis=-1)
        batch_index = tf.expand_dims(
            tf.tile(batch_index_one, [1, self._msize1, self._msize2]),
            axis=-1)
        dpool_index_ex = tf.concat([batch_index, dpool_index], axis=3)
        x_expand = tf.gather_nd(x, dpool_index_ex)
        stride1 = self._msize1 // self._psize1
        stride2 = self._msize2 // self._psize2

        x_pool = tf.nn.max_pool(x_expand,
                                [1, stride1, stride2, 1],
                                [1, stride1, stride2, 1],
                                "VALID")
        return x_pool

    def compute_output_shape(self, input_shape: list) -> tuple:
        """
        Calculate the layer output shape.

        :param input_shape: the shapes of the input tensors,
            for DynamicPoolingLayer we need tow input tensors.
        """
        input_shape_one = input_shape[0]
        return (None, self._psize1, self._psize2, input_shape_one[3])

    def get_config(self) -> dict:
        """Get the config dict of DynamicPoolingLayer."""
        config = {
            'psize1': self._psize1,
            'psize2': self._psize2
        }
        base_config = super(DynamicPoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _validate_dpool_size(self):
        suggestion = self.get_size_suggestion(
            self._msize1, self._msize2, self._psize1, self._psize2
        )
        if suggestion != (self._psize1, self._psize2):
            raise ValueError(
                "DynamicPooling Layer can not "
                f"generate ({self._psize1} x {self._psize2}) output "
                f"feature map, please use ({suggestion[0]} x {suggestion[1]})"
                f" instead. `model.params['dpool_size'] = {suggestion}` "
            )

    @classmethod
    def get_size_suggestion(
        cls,
        msize1: int,
        msize2: int,
        psize1: int,
        psize2: int
    ) -> typing.Tuple[int, int]:
        """
        Get `dpool_size` suggestion for a given shape.

        Returns the nearest legal `dpool_size` for the given combination of
        `(psize1, psize2)`.

        :param msize1: size of the left text.
        :param msize2: size of the right text.
        :param psize1: base size of the pool.
        :param psize2: base size of the pool.
        :return:
        """
        stride1 = msize1 // psize1
        stride2 = msize2 // psize2
        suggestion1 = msize1 // stride1
        suggestion2 = msize2 // stride2
        return (suggestion1, suggestion2)
