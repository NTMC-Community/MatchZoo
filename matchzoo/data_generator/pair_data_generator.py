"""Pair-wise data generator."""

import math

import numpy as np
import pandas as pd

from matchzoo.data_pack import DataPack
from matchzoo.data_generator import DataGenerator


class PairDataGenerator(DataGenerator):
    """
    Generate pair-wise data.

    Examples:
        >>> np.random.seed(111)
        >>> import matchzoo as mz
        >>> input = mz.datasets.toy.load_train_rank_data()
        >>> data_generator = PairDataGenerator(input, 2, 1, 1, False)
        >>> data_generator.num_instance
        2
        >>> len(data_generator)
        2
        >>> x, y = data_generator[0]
        >>> x['id_left'].tolist()
        ['qid0', 'qid0']
        >>> x['text_left'].tolist()
        ['how are glacier caves formed ?', 'how are glacier caves formed ?']
        >>> x['id_right'].tolist()
        ['did3', 'did1']
        >>> x['text_right'].tolist()
        ['A glacier cave is a cave formed within the ice of a glacier .', 'The\
 ice facade is approximately 60 m high']
        >>> y.tolist()
        [[1.0], [0.0]]

    """

    def __init__(self, data_pack: DataPack, num_dup: int = 1, num_neg: int = 1,
                 batch_size: int = 32, shuffle: bool = True):
        """:class:`PairDataGenerator` constructor.

        :param num_dup: number of duplicates for each positive sample.
        :param num_neg: number of negative samples associated with each
            positive sample.
        """
        self._steps = num_neg + 1
        self._data_pack = self.reorganize_data_pack(data_pack,
                                                    num_dup,
                                                    num_neg)
        # Here the super().__init_ must be after the self._data_pack
        super().__init__(self._data_pack, batch_size, shuffle)

    @property
    def num_instance(self) -> int:
        """Get the total number of pairs."""
        return int(math.ceil(len(self._data_pack) / self._steps))

    def _get_batch_of_transformed_samples(self, indices: np.array):
        """Get a batch of paired instances."""
        paired_indices = []
        for index in indices:
            paired_indices.extend(
                range(index * self._steps, (index + 1) * self._steps))
        return self._data_pack[paired_indices].unpack()

    @classmethod
    def reorganize_data_pack(cls, data_pack: DataPack, num_dup: int = 1,
                             num_neg: int = 1):
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
        return DataPack(relation=new_relation,
                        left=data_pack.left.copy(),
                        right=data_pack.right.copy())
