"""Base generator."""

import math
import abc
import typing

import keras
import numpy as np


class BaseGenerator(keras.utils.Sequence):
    """Abstract base class of all matchzoo generators.

    Every generator must implement :meth:`_get_batch_of_transformed_samples`
    method.

    """

    def __init__(
        self,
        batch_size: int = 32,
        num_instances: int = 0,
        shuffle: bool = True
    ):
        """
        :class:`BaseGenerator` constructor.

        :param batch_size: number of instances for each batch
        :param num_instances: total number of instances
        :param shuffle: a bool variable to determine whether choose samples
        randomly
        """
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_instances = num_instances
        self._index_array = None
        self._set_index_array()

    def _set_index_array(self):
        """
        Set the :attr:`index_array`.

        Here the :attr:`index_array` records the index of all the instances.
        """
        if self._shuffle:
            self._index_array = np.random.permutation(self._num_instances)
        else:
            self._index_array = np.arange(self._num_instances)

    def __getitem__(self, idx: int) -> typing.Tuple[dict, list]:
        """Get a batch from index idx.

        :param idx: the index of the batch.
        """
        if idx >= len(self):
            msg = f'Asked to retrieve element {idx}, '
            msg += f'but the Sequence has length {len(self)}'
            raise ValueError(msg)
        if idx == len(self) - 1:
            index_array = self._index_array[self._batch_size * idx:]
        else:
            lower = self._batch_size * idx
            upper = self._batch_size * (idx + 1)
            index_array = self._index_array[lower:upper]
        return self._get_batch_of_transformed_samples(index_array)

    def __len__(self) -> int:
        """Get the total number of batches."""
        return math.ceil(self._num_instances / self._batch_size)

    def on_epoch_end(self):
        """Reorganize the index array while epoch is ended."""
        self._set_index_array()

    def reset(self):
        """Reset the generator from begin."""
        self._set_index_array()

    @abc.abstractmethod
    def _get_batch_of_transformed_samples(
        self,
        index_array: np.array
    ) -> typing.Tuple[dict, list]:
        """Get a batch of transformed samples.

        :param index_array: Arrray of sample indices to include in a batch.
        """
