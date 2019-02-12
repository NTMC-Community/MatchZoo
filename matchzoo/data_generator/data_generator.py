"""Base generator."""

import math
import typing

import keras
import numpy as np
import pandas as pd

import matchzoo as mz
from matchzoo.data_generator.callbacks import Callback


class DataGenerator(keras.utils.Sequence):
    """
    Data Generator.

    Used to divide a :class:`matchzoo.DataPack` into batches. This is helpful
    for generating batch-wise features and delaying data preprocessing to the
    `fit` time.

    See `tutorials/data_handling.ipynb` for a walkthrough.

    :param data_pack: DataPack to generator data from.
    :param mode: One of "point", "pair", and "list". (default: "point")
    :param num_dup: Number of duplications per instance, only effective when
        `mode` is "pair". (default: 1)
    :param num_neg: Number of negative samples per instance, only effective
        when `mode` is "pair". (default: 1)
    :param resample: Either to resample for each epoch, only effective when
        `mode` is "pair". (default: `True`)
    :param batch_size: Batch size. (default: 128)
    :param shuffle: Either to shuffle the samples/instances. (default: `True`)
    :param callbacks: Callbacks. See `matchzoo.data_generator.callbacks` for
        more details.

    Examples::
        >>> import numpy as np
        >>> import matchzoo as mz
        >>> np.random.seed(0)
        >>> data_pack = mz.datasets.toy.load_data()
        >>> batch_size = 8

    To generate data points:
        >>> point_gen = mz.DataGenerator(
        ...     data_pack=data_pack,
        ...     batch_size=batch_size
        ... )
        >>> len(point_gen)
        13
        >>> x, y = point_gen[0]
        >>> for key, value in sorted(x.items()):
        ...     print(key, str(value)[:30])
        id_left ['Q6' 'Q17' 'Q1' 'Q13' 'Q16' '
        id_right ['D6-6' 'D17-1' 'D1-2' 'D13-3'
        text_left ['how long is the term for fed
        text_right ['See Article I and Article II

    To generate data pairs:
        >>> pair_gen = mz.DataGenerator(
        ...     data_pack=data_pack,
        ...     mode='pair',
        ...     num_dup=4,
        ...     num_neg=4,
        ...     batch_size=batch_size,
        ...     shuffle=False
        ... )
        >>> len(pair_gen)
        3
        >>> x, y = pair_gen[0]
        >>> for key, value in sorted(x.items()):
        ...     print(key, str(value)[:30])
        id_left ['Q1' 'Q1' 'Q1' 'Q1' 'Q1' 'Q1'
        id_right ['D1-3' 'D1-4' 'D1-0' 'D1-1' '
        text_left ['how are glacier caves formed
        text_right ['A glacier cave is a cave for

    To generate data lists:
        # TODO:

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
            raise ValueError(f"{mode} is not a valid mode type."
                             f"Must be one of `point`, `pair` or `list`.")

        self._mode = mode
        self._num_dup = num_dup
        self._num_neg = num_neg
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._resample = resample
        self._orig_relation = data_pack.relation
        self._callbacks = callbacks

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
        self._handle_callbacks_on_batch_unpacked(x, y)
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
        # index pool: index -> instance index
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
            raise NotImplementedError(
                f'{self._mode} data generator not implemented.')
        else:
            raise ValueError(f"{self._mode} is not a valid mode type"
                             f"Must be one of `point`, `pair` or `list`.")

        if self._shuffle:
            np.random.shuffle(index_pool)

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

    def _handle_callbacks_on_batch_unpacked(self, x, y):
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

    @property
    def batch_indices(self):
        """`batch_indices` getter."""
        return self._batch_indices

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
