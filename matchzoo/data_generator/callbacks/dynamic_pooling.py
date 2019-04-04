import numpy as np

from matchzoo.data_generator.callbacks import Callback


class DynamicPooling(Callback):
    """:class:`DPoolPairDataGenerator` constructor.

    :param fixed_length_left: max length of left text.
    :param fixed_length_right: max length of right text.
    :param compress_ratio_left: the length change ratio,
        especially after normal pooling layers.
    :param compress_ratio_right: the length change ratio,
        especially after normal pooling layers.
    """

    def __init__(
        self,
        fixed_length_left: int,
        fixed_length_right: int,
        compress_ratio_left: float = 1,
        compress_ratio_right: float = 1,
    ):
        """Init."""
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._compress_ratio_left = compress_ratio_left
        self._compress_ratio_right = compress_ratio_right

    def on_batch_unpacked(self, x, y):
        """
        Insert `dpool_index` into `x`.

        :param x: unpacked x.
        :param y: unpacked y.
        """
        x['dpool_index'] = _dynamic_pooling_index(
            x['length_left'],
            x['length_right'],
            self._fixed_length_left,
            self._fixed_length_right,
            self._compress_ratio_left,
            self._compress_ratio_right
        )


def _dynamic_pooling_index(length_left: np.array,
                           length_right: np.array,
                           fixed_length_left: int,
                           fixed_length_right: int,
                           compress_ratio_left: float,
                           compress_ratio_right: float) -> np.array:
    def _dpool_index(one_length_left: int,
                     one_length_right: int,
                     fixed_length_left: int,
                     fixed_length_right: int):
        if one_length_left == 0:
            stride_left = fixed_length_left
        else:
            stride_left = 1.0 * fixed_length_left / one_length_left

        if one_length_right == 0:
            stride_right = fixed_length_right
        else:
            stride_right = 1.0 * fixed_length_right / one_length_right

        one_idx_left = [int(i / stride_left)
                        for i in range(fixed_length_left)]
        one_idx_right = [int(i / stride_right)
                         for i in range(fixed_length_right)]
        mesh1, mesh2 = np.meshgrid(one_idx_left, one_idx_right)
        index_one = np.transpose(
            np.stack([mesh1, mesh2]), (2, 1, 0))
        return index_one

    index = []
    dpool_bias_left = dpool_bias_right = 0
    if fixed_length_left % compress_ratio_left != 0:
        dpool_bias_left = 1
    if fixed_length_right % compress_ratio_right != 0:
        dpool_bias_right = 1
    cur_fixed_length_left = int(
        fixed_length_left // compress_ratio_left) + dpool_bias_left
    cur_fixed_length_right = int(
        fixed_length_right // compress_ratio_right) + dpool_bias_right
    for i in range(len(length_left)):
        index.append(_dpool_index(
            length_left[i] // compress_ratio_left,
            length_right[i] // compress_ratio_right,
            cur_fixed_length_left,
            cur_fixed_length_right))
    return np.array(index)
