"""Base generator."""

import math
import typing

import keras
import numpy as np

from matchzoo import upsample


class DataGenerator(keras.utils.Sequence):
    """Abstract base class of all matchzoo generators.

    Every generator must implement :meth:`_get_batch_of_transformed_samples`
    method.

    """

    def __init__(
        self,
        data_pack,
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """
        :class:`DataGenerator` constructor.

        :param batch_size: number of instances for each batch
        :param shuffle: a bool variable to determine whether choose samples
        randomly
        """
        self._data_pack = data_pack
        self._batch_size = batch_size
        self._shuffle = shuffle

        self._indices = None
        self._set_indices()

    def __getitem__(self, idx: int) -> typing.Tuple[dict, list]:
        """Get a batch from index idx.

        :param idx: the index of the batch.
        """
        lower = self._batch_size * idx
        upper = self._batch_size * (idx + 1)
        indices = self._indices[lower:upper]
        return self._get_batch_of_transformed_samples(indices)

    def _get_batch_of_transformed_samples(
        self,
        indices: np.array
    ) -> typing.Tuple[dict, typing.Any]:
        """Get a batch of samples based on their ids.

        :param indices: a list of instance ids.
        :return: A batch of transformed samples.
        """
        return self._data_pack[indices].unpack()

    def __len__(self) -> int:
        """Get the total number of batches."""
        return math.ceil(len(self._data_pack) / self._batch_size)

    def on_epoch_end(self):
        """Reorganize the index array while epoch is ended."""
        self._set_indices()

    def reset(self):
        """Reset the generator from begin."""
        self._set_indices()

    def _set_indices(self):
        """
        Set the :attr:`index_array`.

        Here the :attr:`index_array` records the index of all the instances.
        """
        num_instances = len(self._data_pack)
        if self._shuffle:
            self._indices = np.random.permutation(num_instances)
        else:
            self._indices = np.arange(num_instances)


class UnitPostProcessDataGenerator(DataGenerator):
    def __init__(self, *args, unit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._unit = unit

    def _get_batch_of_transformed_samples(self, indices: np.array):
        dp = self._data_pack[indices].copy()
        func = self._unit.transform
        dp.left['text_left'] = dp.left['text_left'].apply(func)
        dp.right['text_right'] = dp.right['text_right'].apply(func)
        return dp.unpack()


class PostProcessDataGenerator(DataGenerator):
    def __init__(self, *args, func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._func = func

    def _get_batch_of_transformed_samples(self, indices: np.array):
        dp = self._data_pack[indices].copy()
        dp.left['text_left'] = dp.left['text_left'].apply(self._func)
        dp.right['text_right'] = dp.right['text_right'].apply(self._func)
        return dp.unpack()


class OrigPairGeneratorUsingNewInterface(DataGenerator):
    def __init__(self, *args, num_neg=1, num_dup=4, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_dup = num_dup
        self._num_neg = num_neg

    def _get_batch_of_transformed_samples(self, indices):
        return upsample(self._data_pack[indices],
                        num_dup=self._num_dup,
                        num_neg=self._num_neg).unpack()


class DataGeneratorFusion(DataGenerator):
    def __init__(self, *args, num_neg=1, num_dup=1, unit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._unit = unit
        self._num_dup = num_dup
        self._num_neg = num_neg

    def _get_batch_of_transformed_samples(self, indices):
        dp = self._data_pack[indices].copy()
        func = self._unit.transform
        dp.left['text_left'] = dp.left['text_left'].apply(func)
        dp.right['text_right'] = dp.right['text_right'].apply(func)
        return upsample(dp, num_dup=self._num_dup,
                        num_neg=self._num_neg).unpack()
