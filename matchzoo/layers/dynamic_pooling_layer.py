"""An implementation of Dynamic Pooling Layer."""
import typing

from keras import backend as K
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
        x, dpool_index = inputs
        x_expand = K.tf.gather_nd(x, dpool_index)
        stride1 = self._msize1 / self._psize1
        stride2 = self._msize2 / self._psize2

        suggestion1 = self._msize1 / stride1
        suggestion2 = self._msize2 / stride2

        if suggestion1 != self._psize1 or suggestion2 != self._psize2:
            raise ValueError("DynamicPooling Layer can not "
                             "generate ({} x {}) output feature map, "
                             "please use ({} x {} instead.)"
                             .format(self._psize1, self._psize2,
                                     suggestion1, suggestion2))

        x_pool = K.tf.nn.max_pool(x_expand,
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
