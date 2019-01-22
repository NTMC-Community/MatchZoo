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
        mode='point',
        num_dup: int = 1,
        num_neg: int = 1,
        resample: bool = True,
        batch_size: int = 128,
        shuffle: bool = True,
        callbacks: typing.List[Callback] = None
    ):
        """Init."""
        if callbacks is None:
            callbacks = []

        if mode not in ('point', 'pair', 'list'):
            raise ValueError

        self._callbacks = callbacks
        self._mode = mode
        self._num_dup = num_dup
        self._num_neg = num_neg
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._resample = resample
        self._orig_relation = data_pack.relation

        if mode == 'pair':
            data_pack.relation = self._reorganize_pair_wise(
                data_pack.relation,
                num_dup=num_dup,
                num_neg=num_neg
            )

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
        if self._mode == 'pair' and self._resample:
            self._data_pack.relation = self._reorganize_pair_wise(
                relation=self._orig_relation,
                num_dup=self._num_dup,
                num_neg=self._num_neg
            )
        self.reset_index()

    def reset_index(self):
        """
        Set the :attr:`index_array`.

        Here the :attr:`index_array` records the index of all the instances.
        """
        # index pool: index -> instance
        if self._mode == 'point':
            num_instances = len(self._data_pack)
            index_pool = list(range(num_instances))
        elif self._mode == 'pair':
            index_pool = []
            step_size = self._num_neg + 1
            num_instances = int(len(self._data_pack) / step_size)
            for i in range(num_instances):
                lower = i * step_size
                upper = (i + 1) * step_size
                indices = list(range(lower, upper))
                if indices:
                    index_pool.append(indices)
        elif self._mode == 'list':
            raise NotImplementedError
        else:
            raise ValueError

        if self._shuffle:
            random.shuffle(index_pool)

        # batch_indices: index -> batch of indices
        self._batch_indices = []
        for i in range(math.ceil(num_instances / self._batch_size)):
            lower = self._batch_size * i
            upper = self._batch_size * (i + 1)
            candidates = index_pool[lower:upper]
            if self._mode == 'pair':
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
    def mode(self):
        """`mode` getter."""
        return self._mode

    @mode.setter
    def mode(self, value):
        """`mode` setter."""
        self._mode = value
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
        relation: pd.DataFrame,
        num_dup: int = 1,
        num_neg: int = 1
    ):
        """Re-organize the data pack as pair-wise format."""
        pairs = []
        groups = relation.sort_values(
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
        return new_relation
