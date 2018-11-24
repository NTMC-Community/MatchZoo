"""Base generator."""

import math
import typing
import numpy as np

from matchzoo import reorganize_data_pack_pair_wise
from .data_generator import DataGenerator


class PairDataGenerator(DataGenerator):
    """Generate pair-wise data...."""
    def __init__(self, data_pack, num_dup: int=1, num_neg: int=1, **kwargs):
        """:class:`PairDataGenerator` constructor...."""
        self._num_dup = num_dup
        self._num_neg = num_neg
        self._data_pack = reorganize_data_pack_pair_wise(data_pack, num_dup,
                                                         num_neg)
        # Here the super().__init_ must be after the self._data_pack
        super().__init__(self._data_pack, **kwargs)

    @property
    def num_instance(self) -> int:
        """Get the total number of pairs...."""
        return math.ceil(len(self._data_pack) / (self._num_neg + 1))

    def _get_batch_of_transformed_samples(self, indices: np.array):
        """Get a batch of paired instances...."""
        paired_indices = []
        steps = self._num_neg + 1
        for index in indices:
            paired_indices.extend(
                list(range(index * steps, (index + 1) * steps)))
        return self._data_pack[paired_indices].unpack()
