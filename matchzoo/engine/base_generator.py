"""Base generator."""

import abc
import keras
import numpy as np


class BaseGenerator(keras.utils.Sequence):
    """Abstract base generator of all matchzoo generators.

    Every generator must implement :meth:`_get_batch_of_transformed_samples`
    method.

    """

    def __init__(self, batch_size: int=32, num_instances: int=0, shuffle:
                 bool=True):
        """
        :class:`BaseGenerator` constructor.

        :param batch_size: number of instances for each batch
        :param num_instances: total number of instances
        :param shuffle: a bool variable to determine whether choose samples
        randomly
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_instances = num_instances
        self.batch_index = 0
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        """Set the :attr:`index_array`.

        Here the :attr:`index_array` records the index of all the instances.
        """
        self.index_array = np.arange(self.num_instances)
        if self.shuffle:
            self.index_array = np.random.permutation(self.num_instances)

    def __getitem__(self, idx):
        """Get a batch from index idx.

        :param idx: the index of the batch.
        """
        if idx >= len(self):
            msg = f'Asked to retrieve element {idx}, '
            msg += f'but the Sequence has length {len(self)}'
            raise ValueError(msg)
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batch_of_transformed_samples(index_array)

    def __len__(self):
        """Get the total number of batches."""
        return (self.num_instances + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        """Reorganize the index array while epoch is ended."""
        self._set_index_array()

    def reset(self):
        """Reset the batch_index to generator from begin."""
        self.batch_index = 0

    def _flow_index(self):
        """Yield a batch of sample indices."""
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.batch_index == 0:
                self._set_index_array()
            curr_index = (self.batch_index * self.batch_size) % \
                self.num_instances
            # note here, some samples may be missed.
            if self.num_instances > curr_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            yield self.index_array[curr_index: curr_index + self.batch_size]

    @abc.abstractmethod
    def _get_batch_of_transformed_samples(self, index_array):
        """Get a batch of transformed samples.

        :param index_array: Arrray of sample indices to include in a batch.

        :return: A batch of transformed samples.

        """
