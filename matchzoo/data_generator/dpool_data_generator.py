"""Data generator with dynamic pooling."""

import math

import numpy as np
import pandas as pd

from matchzoo.data_pack import DataPack
from matchzoo.data_generator import DataGenerator
from matchzoo.data_generator import PairDataGenerator


def _dynamic_pooling_index(length_left: np.array,
                           length_right: np.array,
                           fixed_length_left: int,
                           fixed_length_right: int,
                           compress_ratio_left: float,
                           compress_ratio_right: float) -> np.array:

    def _dpool_index(batch_idx: int,
                     one_length_left: int,
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
            np.stack([np.ones(mesh1.shape) * batch_idx,
                      mesh1, mesh2]), (2, 1, 0))
        return index_one

    index = []
    dpool_bias_left = dpool_bias_right = 0
    if fixed_length_left % compress_ratio_left != 0:
        dpool_bias_left = 1
    if fixed_length_right % compress_ratio_right != 0:
        dpool_bias_right = 1
    cur_fixed_length_left = fixed_length_left // compress_ratio_left \
        + dpool_bias_left
    cur_fixed_length_right = fixed_length_right // compress_ratio_right \
        + dpool_bias_right
    for i in range(len(length_left)):
        index.append(_dpool_index(i,
                                  length_left[i] // compress_ratio_left,
                                  length_right[i] // compress_ratio_right,
                                  cur_fixed_length_left,
                                  cur_fixed_length_right))
    return np.array(index)


class DPoolDataGenerator(DataGenerator):
    """
    Generate data with dynamic pooling.

    Examples:
        >>> import matchzoo as mz
        >>> raw_data = mz.datasets.toy.load_data()
        >>> preprocessor = mz.preprocessors.BasicPreprocessor(
        ...     fixed_length_left=10,
        ...     fixed_length_right=40,
        ...     remove_stop_words=True)
        >>> processed_data = preprocessor.fit_transform(raw_data)
        >>> data_generator = DPoolDataGenerator(processed_data, 3, 10,
        ...     batch_size=3, shuffle=False)
        >>> len(data_generator)
        34
        >>> data_generator.num_instance
        100
        >>> x, y = data_generator[-1]
        >>> type(x)
        <class 'dict'>
        >>> x.keys()
        dict_keys(['id_left', 'text_left', 'length_left', \
'id_right', 'text_right', 'length_right', 'dpool_index'])
        >>> type(x['id_left'])
        <class 'numpy.ndarray'>
        >>> type(x['id_right'])
        <class 'numpy.ndarray'>
        >>> type(x['text_left'])
        <class 'numpy.ndarray'>
        >>> type(x['text_right'])
        <class 'numpy.ndarray'>
        >>> type(y)
        <class 'numpy.ndarray'>

    """

    def __init__(self,
                 data_pack: DataPack,
                 fixed_length_left: int,
                 fixed_length_right: int,
                 compress_ratio_left: float = 1,
                 compress_ratio_right: float = 1,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """:class:`DPoolDataGenerator` constructor.

        :param fixed_length_left: max length of left text.
        :param fixed_length_right: max length of right text.
        :param compress_ratio_left: the length change ratio,
            especially after normal pooling layers.
        :param compress_ratio_right: the length change ratio,
            especially after normal pooling layers.
        """
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._compress_ratio_left = compress_ratio_left
        self._compress_ratio_right = compress_ratio_right
        # Here the super().__init_ must be after the self._data_pack
        super().__init__(data_pack, batch_size, shuffle)

    def _get_batch_of_transformed_samples(self, indices: np.array):
        """Get a batch of instances."""
        x, y = super()._get_batch_of_transformed_samples(indices)
        x['dpool_index'] = _dynamic_pooling_index(
            x['length_left'],
            x['length_right'],
            self._fixed_length_left,
            self._fixed_length_right,
            self._compress_ratio_left,
            self._compress_ratio_right
        )
        return (x, y)


class DPoolPairDataGenerator(PairDataGenerator):
    """
    Generate pair-wise data with dynamic pooling.

    Examples:
        >>> np.random.seed(111)
        >>> import matchzoo as mz
        >>> raw_data = mz.datasets.toy.load_data()
        >>> preprocessor = mz.preprocessors.BasicPreprocessor(
        ...     fixed_length_left=10,
        ...     fixed_length_right=40,
        ...     remove_stop_words=True)
        >>> processed_data = preprocessor.fit_transform(raw_data)
        >>> data_generator = DPoolPairDataGenerator(processed_data, 3, 10,
        ...     1, 1, 2, 1, 3, False)
        >>> data_generator.num_instance
        10
        >>> len(data_generator)
        4
        >>> x, y = data_generator[0]
        >>> type(x)
        <class 'dict'>
        >>> x.keys()
        dict_keys(['id_left', 'text_left', 'length_left', \
'id_right', 'text_right', 'length_right', 'dpool_index'])
        >>> type(x['id_left'])
        <class 'numpy.ndarray'>
        >>> type(x['id_right'])
        <class 'numpy.ndarray'>
        >>> type(x['text_left'])
        <class 'numpy.ndarray'>
        >>> type(x['text_right'])
        <class 'numpy.ndarray'>
        >>> len(x['id_left'])
        6
        >>> len(x['id_right'])
        6
        >>> type(y)
        <class 'numpy.ndarray'>

    """

    def __init__(self,
                 data_pack: DataPack,
                 fixed_length_left: int,
                 fixed_length_right: int,
                 compress_ratio_left: float = 1,
                 compress_ratio_right: float = 1,
                 num_dup: int = 1,
                 num_neg: int = 1,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """:class:`DPoolPairDataGenerator` constructor.

        :param fixed_length_left: max length of left text.
        :param fixed_length_right: max length of right text.
        :param compress_ratio_left: the length change ratio,
            especially after normal pooling layers.
        :param compress_ratio_right: the length change ratio,
            especially after normal pooling layers.
        """
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._compress_ratio_left = compress_ratio_left
        self._compress_ratio_right = compress_ratio_right
        # Here the super().__init__ must be after the self._data_pack
        super().__init__(data_pack, num_dup, num_neg, batch_size, shuffle)

    def _get_batch_of_transformed_samples(self, indices: np.array):
        """Get a batch of paired instances."""
        x, y = super()._get_batch_of_transformed_samples(indices)
        x['dpool_index'] = _dynamic_pooling_index(
            x['length_left'],
            x['length_right'],
            self._fixed_length_left,
            self._fixed_length_right,
            self._compress_ratio_left,
            self._compress_ratio_right
        )
        return (x, y)
