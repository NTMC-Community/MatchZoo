"""Matchzoo point generator."""

from matchzoo import engine
from matchzoo import tasks
from matchzoo import datapack
from matchzoo import utils

import numpy as np
import typing


class PointGenerator(engine.BaseGenerator):
    """PointGenerator for Matchzoo.

    Ponit generator can be used for classification as well as ranking.

    Examples:
        >>> import pandas as pd
        >>> relation = [['qid0', 'did0', 1]]
        >>> left_data = [['qid0', [1, 2]]]
        >>> right_data = [['did0', [2, 3]]]
        >>> relation_columns = ['id_left', 'id_right', 'label']
        >>> left_columns = ['id_left', 'text_left']
        >>> right_columns = ['id_right', 'text_right']
        >>> relation_df = pd.DataFrame(relation, columns=relation_columns)
        >>> left_df = pd.DataFrame(left_data, columns=left_columns)
        >>> left_df.set_index('id_left', inplace=True)
        >>> right_df = pd.DataFrame(right_data, columns=right_columns)
        >>> right_df.set_index('id_right', inplace=True)
        >>> input = datapack.DataPack(relation=relation_df,
        ...                           left_data=left_df,
        ...                           right_data=right_df
        ... )
        >>> task = tasks.Classification(num_classes=2)
        >>> from matchzoo.generators import PointGenerator
        >>> generator = PointGenerator(input, task, 1, 'train', False)
        >>> x, y = generator[0]
        >>> assert x['text_left'].tolist() == [[1, 2]]
        >>> assert x['text_right'].tolist() == [[2, 3]]
        >>> assert y.tolist() == [[0., 1.]]

    """

    def __init__(
        self,
        inputs: datapack.DataPack,
        task: engine.BaseTask=tasks.Classification(2),
        batch_size: int=32,
        stage: str='train',
        shuffle: bool=True
    ):
        """Construct the point generator.

        :param inputs: the output generated by :class:`DataPack`.
        :param task: the task is a instance of :class:`engine.BaseTask`.
        :param batch_size: number of instances in a batch.
        :param shuffle: whether to shuffle the instances while generating a
            batch.
        """
        self._task = task
        self.data = self.transform_data(inputs)
        self.left_data = inputs.left_data
        self.right_data = inputs.right_data
        super().__init__(batch_size, len(inputs.relation), stage, shuffle)

    def transform_data(self, inputs: datapack.DataPack) -> dict:
        """Obtain the transformed data from :class:`DataPack`.

        :param inputs: An instance of :class:`DataPack` to be transformed.
        :return: the output of all the transformed inputs.
        """
        relation = inputs.relation
        out = {}
        for column in relation.columns:
            out[column] = np.asarray(relation[column])
        return out

    def _get_batch_of_transformed_samples(
        self,
        index_array: list
    ) -> typing.Tuple[dict, typing.Any]:
        """Get a batch of samples based on their ids.

        :param index_array: a list of instance ids.
        :return: A batch of transformed samples.
        """
        bsize = len(index_array)
        batch_x = {}
        batch_y = None
        if self.stage == 'train':
            if isinstance(self._task, tasks.Ranking):
                batch_y = map(self._task.output_dtype, self.data['label'])
            elif isinstance(self._task, tasks.Classification):
                batch_y = np.zeros((bsize, self._task.num_classes))
                for idx, label in enumerate(self.data['label'][index_array]):
                    label = self._task.output_dtype(label)
                    batch_y[idx, label] = 1
            else:
                msg = f"{self._task} is not a valid task type."
                msg += ":class:`Ranking` and :class:`Classification` expected."
                raise ValueError(msg)
        batch_x['ids'] = []
        columns = self.left_data.columns.values.tolist() + \
            self.right_data.columns.values.tolist()
        for column in columns:
            batch_x[column] = []
        for idx in index_array:
            id_left = self.data['id_left'][idx]
            id_right = self.data['id_right'][idx]
            for column in self.left_data.columns:
                val = self.left_data.loc[id_left, column]
                batch_x[column].append(val)
            for column in self.right_data.columns:
                val = self.right_data.loc[id_right, column]
                batch_x[column].append(val)
            batch_x['ids'].append([id_left, id_right])
        for key, val in batch_x.items():
            batch_x[key] = np.array(val)
        batch_x = utils.dotdict(batch_x)
        return (batch_x, batch_y)
