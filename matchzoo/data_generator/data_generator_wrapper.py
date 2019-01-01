"""Data generator wrappers."""

import numpy as np

from matchzoo.data_generator import DataGenerator


def dynamic_pooling_wrapper(generator: DataGenerator,
                            fixed_length_left: int,
                            fixed_length_right: int,
                            compress_ratio_left: float = 1,
                            compress_ratio_right: float = 1):
    """Dynamic Pooling Data Generator Wrapper."""

    def _dynamic_pooling_index(length_left: np.array,
                               length_right: np.array,
                               fixed_length_left: int,
                               fixed_length_right: int,
                               compress_ratio_left: float,
                               compress_ratio_right: float):

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

    def _wrapper(generator, indices: np.array):
        feed_data = generator._get_batch_of_transformed_samples(indices)
        feed_data['dpool_index'] = _dynamic_pooling_index(
            feed_data['length_left'],
            feed_data['length_right'],
            fixed_length_left,
            fixed_length_right,
            compress_ratio_left,
            compress_ratio_right
        )
        return feed_data

    methodes = set(dir(generator))
    if '_get_batch_of_transformed_samples' not in methodes:
        return generator
    # if 'length_left' not in generator.header?
    setattr(generator, '_get_batch_of_transformed_samples', _wrapper)
    return generator
