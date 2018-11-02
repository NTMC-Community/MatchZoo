"""Matchzoo point generator."""

from matchzoo import engine
from matchzoo import tasks
from matchzoo import datapack
from matchzoo import utils
from matchzoo import preprocessor

import numpy as np
import typing


class PointGenerator(engine.BaseGenerator):
    """PointGenerator for Matchzoo.

    Ponit generator can be used for classification as well as ranking.

    Examples:
        >>> import pandas as pd
        >>> relation = [['qid0', 'did0', 1]]
        >>> left = [['qid0', [1, 2]]]
        >>> right = [['did0', [2, 3]]]
        >>> relation = pd.DataFrame(relation,
        ...                         columns=['id_left', 'id_right', 'label'])
        >>> left = pd.DataFrame(left, columns=['id_left', 'text_left'])
        >>> left.set_index('id_left', inplace=True)
        >>> right = pd.DataFrame(right, columns=['id_right', 'text_right'])
        >>> right.set_index('id_right', inplace=True)
        >>> input = datapack.DataPack(relation=relation,
        ...                           left=left,
        ...                           right=right
        ... )
        >>> task = tasks.Classification()
        >>> generator = PointGenerator(input, task, 1, 'train', False)
        >>> x, y = generator[0]
        >>> x['text_left'].tolist()
        [[1, 2]]
        >>> x['text_right'].tolist()
        [[2, 3]]
        >>> x['id_left'].tolist()
        ['qid0']
        >>> x['id_right'].tolist()
        ['did0']
        >>> y.tolist()
        [[0.0, 1.0]]

    """

    def __init__(
        self,
        inputs: datapack.DataPack,
        task: engine.BaseTask = tasks.Classification(2),
        batch_size: int = 32,
        stage: str = 'train',
        shuffle: bool = True,
        use_word_hashing: bool = False
    ):
        """Construct the point generator.

        :param inputs: the output generated by :class:`DataPack`.
        :param task: the task is a instance of :class:`engine.BaseTask`.
        :param batch_size: number of instances in a batch.
        :param stage: String indicate the pre-processing stage, `train`,
            `evaluate`, or `predict` expected.
        :param shuffle: whether to shuffle the instances while generating a
            batch.
        """
        self._relation = inputs.relation
        self._task = task
        self._left = inputs.left
        self._right = inputs.right
        self._context = inputs.context
        self._use_word_hashing = use_word_hashing
        super().__init__(batch_size, len(inputs.relation), stage, shuffle)

    def _get_batch_of_transformed_samples(
        self,
        index_array: np.array
    ) -> typing.Tuple[dict, typing.Any]:
        """Get a batch of samples based on their ids.

        :param index_array: a list of instance ids.
        :return: A batch of transformed samples.
        """
        batch_x = {}
        batch_y = None

        columns = self._left.columns.values.tolist() + \
            self._right.columns.values.tolist() + ['id_left', 'id_right']
        for column in columns:
            batch_x[column] = []

        # Create label field.
        if self.stage in ['train', 'evaluate']:
            if isinstance(self._task, tasks.Ranking):
                self._relation['label'] = self._relation['label'].astype(
                    self._task.output_dtype)
                batch_y = self._relation['label'][index_array].values
            elif isinstance(self._task, tasks.Classification):
                self._relation['label'] = self._relation['label'].astype(
                    self._task.output_dtype)
                batch_y = np.zeros((len(index_array), self._task.num_classes))
                for idx, label in enumerate(
                        self._relation['label'][index_array]):
                    batch_y[idx, label] = 1
            else:
                msg = f"{self._task} is not a valid task type."
                msg += ":class:`Ranking` and :class:`Classification` expected."
                raise ValueError(msg)
        # Get batch of X.
        id_left = self._relation.iloc[index_array, 0]
        id_right = self._relation.iloc[index_array, 1]

        batch_x['id_left'] = id_left
        batch_x['id_right'] = id_right

        if self._use_word_hashing:
            self._hash_unit = preprocessor.WordHashingUnit(self._context['term_index'])

        for column in self._left.columns:
            if column == 'text_left' and self._use_word_hashing:
                batch_x[column] = [self._hash_unit.transform(item)
                                   for
                                   item
                                   in
                                   self._left.loc[id_left, column].tolist()]
                continue
            batch_x[column] = self._left.loc[id_left, column].tolist()
        for column in self._right.columns:
            if column == 'text_right' and self._use_word_hashing:
                batch_x[column] = [self._hash_unit.transform(item)
                                   for
                                   item
                                   in
                                   self._right.loc[id_right, column].tolist()]
                continue
            batch_x[column] = self._right.loc[id_right, column].tolist()

        for key, val in batch_x.items():
            batch_x[key] = np.array(val)

        batch_x = utils.dotdict(batch_x)
        return (batch_x, batch_y)
