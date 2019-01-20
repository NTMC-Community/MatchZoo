"""Base generator."""

import math
import typing
import random

import keras
import numpy as np
import pandas as pd

import matchzoo as mz
from matchzoo.data_generator.callbacks import Callback


class DataGenerator(keras.utils.Sequence):
    """Abstract base class of all matchzoo generators.

    Every generator must implement :meth:`_get_batch_of_transformed_samples`
    method.

    Examples:
        >>> import matchzoo as mz
        >>> raw_data = mz.datasets.toy.load_data()
        >>> data_generator = DataGenerator(raw_data, batch_size=3,
        ...                                shuffle=False)
        >>> len(data_generator)
        34
        >>> x, y = data_generator[-1]
        >>> type(x)
        <class 'dict'>
        >>> x.keys()
        dict_keys(['id_left', 'text_left', 'id_right', 'text_right'])
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

    def __init__(
        self,
        data_pack: mz.DataPack,
        pair_wise=False,
        num_dup: int = 1,
        num_neg: int = 1,
        batch_size: int = 128,
        shuffle: bool = True,
        callbacks: typing.List[Callback] = None
    ):
        """
        :class:`DataGenerator` constructor.

        :param data_pack: a :class:`DataPack` object.
        :param batch_size: number of instances for each batch
        :param shuffle: a bool variable to determine whether choose samples
        randomly
        """
        if callbacks is None:
            callbacks = []

        if pair_wise:
            data_pack = self._reorganize_pair_wise(
                data_pack, num_dup=num_dup, num_neg=num_neg)

        self._callbacks = callbacks
        self._pair_wise = pair_wise
        self._num_dup = num_dup
        self._num_neg = num_neg
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._data_pack = data_pack

        self._batch_indices = None
        self.reset_index()

    def __getitem__(self, item: int) -> typing.Tuple[dict, np.ndarray]:
        """Get a batch from index idx.

        :param item: the index of the batch.
        """
        if isinstance(item, slice):
            indices = sum(self._batch_indices[item], [])
        else:
            indices = self._batch_indices[item]
        batch_data_pack = self._data_pack[indices]
        self._handle_callbacks_on_batch_data_pack(batch_data_pack)
        x, y = batch_data_pack.unpack()
        self._handle_callbacks_on_batch_x_y(x, y)
        return x, y

    def __len__(self) -> int:
        """Get the total number of batches."""
        return len(self._batch_indices)

    def on_epoch_end(self):
        """Reorganize the index array while epoch is ended."""
        self.reset_index()

    def reset_index(self):
        """
        Set the :attr:`index_array`.

        Here the :attr:`index_array` records the index of all the instances.
        """
        # index pool: index -> instance
        if self._pair_wise:
            index_pool = []
            step_size = self._num_neg + 1
            num_instances = int(len(self._data_pack) / step_size)
            for i in range(num_instances):
                lower = i * step_size
                upper = (i + 1) * step_size
                indices = list(range(lower, upper))
                if indices:
                    index_pool.append(indices)
        else:
            num_instances = len(self._data_pack)
            index_pool = list(range(num_instances))

        if self._shuffle:
            random.shuffle(index_pool)

        # batch_indices: index -> batch of indices
        self._batch_indices = []
        for i in range(math.ceil(num_instances / self._batch_size)):
            lower = self._batch_size * i
            upper = self._batch_size * (i + 1)
            candidates = index_pool[lower:upper]
            if self._pair_wise:
                candidates = sum(candidates, [])
            if candidates:
                self._batch_indices.append(candidates)

    def _handle_callbacks_on_batch_data_pack(self, batch_data_pack):
        for callback in self._callbacks:
            callback.on_batch_data_pack(batch_data_pack)

    def _handle_callbacks_on_batch_x_y(self, x, y):
        for callback in self._callbacks:
            callback.on_batch_unpacked(x, y)

    @property
    def callbacks(self):
        """`callbacks` getter."""
        return self._callbacks

    @callbacks.setter
    def callbacks(self, value):
        """`callbacks` setter."""
        self._callbacks = value

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        """`num_neg` setter."""
        self._num_neg = value
        self.reset_index()

    @property
    def num_dup(self):
        """`num_dup` getter."""
        return self._num_dup

    @num_dup.setter
    def num_dup(self, value):
        """`num_dup` setter."""
        self._num_dup = value
        self.reset_index()

    @property
    def pair_wise(self):
        """`pair_wise` getter."""
        return self._pair_wise

    @pair_wise.setter
    def pair_wise(self, value):
        """`pair_wise` setter."""
        self._pair_wise = value
        self.reset_index()

    @property
    def batch_size(self):
        """`batch_size` getter."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """`batch_size` setter."""
        self._batch_size = value
        self.reset_index()

    @property
    def shuffle(self):
        """`shuffle` getter."""
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        """`shuffle` setter."""
        self._shuffle = value
        self.reset_index()

    @classmethod
    def _reorganize_pair_wise(
        cls,
        data_pack: mz.DataPack,
        num_dup: int = 1,
        num_neg: int = 1
    ):
        """Re-organize the data pack as pair-wise format.

        :param data_pack: the input :class:`DataPack`.
        :param num_dup: number of duplicates for each positive sample.
        :param num_neg: number of negative samples associated with each
            positive sample.
        :return: the reorganized :class:`DataPack` object.
        """
        pairs = []
        groups = data_pack.relation.sort_values(
            'label', ascending=False).groupby('id_left')
        for idx, group in groups:
            labels = group.label.unique()
            for label in labels[:-1]:
                pos_samples = group[group.label == label]
                pos_samples = pd.concat([pos_samples] * num_dup)
                neg_samples = group[group.label < label]
                for _, pos_sample in pos_samples.iterrows():
                    pos_sample = pd.DataFrame([pos_sample])
                    neg_sample = neg_samples.sample(num_neg, replace=True)
                    pairs.extend((pos_sample, neg_sample))
        new_relation = pd.concat(pairs, ignore_index=True)
        return mz.DataPack(relation=new_relation,
                           left=data_pack.left.copy(),
                           right=data_pack.right.copy())
