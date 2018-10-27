"""Base generator."""

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
        stage: str = 'train',
        shuffle: bool = True
    ):
        """
        :class:`BaseGenerator` constructor.

        :param batch_size: number of instances for each batch
        :param num_instances: total number of instances
        :param stage: String indicate the pre-processing stage, `train`,
            `evaluate`, or `predict` expected.
        :param shuffle: a bool variable to determine whether choose samples
        randomly
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_instances = num_instances
        if stage not in ['train', 'evaluate', 'predict']:
            raise ValueError(f'{stage} is not a valid stage name.')
        self.stage = stage
        self.index_array = None
        self._set_index_array()

    def _set_index_array(self):
        """
        Set the :attr:`index_array`.

        Here the :attr:`index_array` records the index of all the instances.
        """
        if self.shuffle:
            self.index_array = np.random.permutation(self.num_instances)
        else:
            self.index_array = np.arange(self.num_instances)

    def __getitem__(self, idx: int) -> typing.Tuple[dict, list]:
        """Get a batch from index idx.

        :param idx: the index of the batch.
        """
        if idx >= len(self):
            msg = f'Asked to retrieve element {idx}, '
            msg += f'but the Sequence has length {len(self)}'
            raise ValueError(msg)
        if idx == len(self) - 1:
            index_array = self.index_array[self.batch_size * idx:]
        else:
            index_array = self.index_array[self.batch_size * idx:
                                           self.batch_size * (idx + 1)]
        return self._get_batch_of_transformed_samples(index_array)

    def __len__(self) -> int:
        """Get the total number of batches."""
        return (self.num_instances + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> typing.Tuple[dict, list]:
        """Create an generator."""
        if self.stage == 'train':
            while True:
                for item in (self[i] for i in range(len(self))):
                    yield item
        else:
            for item in (self[i] for i in range(len(self))):
                yield item

    def on_epoch_end(self):
        """Reorganize the index array while epoch is ended."""
        self._set_index_array()

    def reset(self):
        """Reset the generator from begin."""
        self._set_index_array()

    @abc.abstractmethod
    def _get_batch_of_transformed_samples(self, index_array: np.array):
        """Get a batch of transformed samples.

        :param index_array: Arrray of sample indices to include in a batch.
        """
