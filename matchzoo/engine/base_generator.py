"""Base generator."""

import abc
import keras
import numpy as np


class BaseGenerator(keras.utils.Sequence):
    """Matchzoo base generator.

    Every `Generator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
    """

    def __init__(self, batch_size: int, shuffle: bool):
        """Initialization."""
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        """Round up."""
        return (self.n + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.batch_index == 0:
                self._set_index_array()
            curr_index = (self.batch_index * self.batch_size) % self.n
            # note here, some samples may be missed.
            if self.n > curr_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            yield self.index_array[curr_index: curr_index + self.batch_size]

    @abc.abstractmethod
    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Arrray of sample indices to include in batch.

        # Returns
            A batch of transformed samples.

        """
