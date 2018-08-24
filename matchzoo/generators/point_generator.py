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
        >>> data = [['qid0', 'did0', 1]]
        >>> columns = ['id_left', 'id_right', 'label']
        >>> features = {'qid0': [1, 2], 'did0': [2, 3]}
        >>> input = datapack.DataPack(data=data,
        ...                           mapping=features,
        ...                           columns=columns
        ...                           )
        >>> task = tasks.Classification(num_classes=2)
        >>> from matchzoo.generators import PointGenerator
        >>> generator = PointGenerator(input, task, 1, 'train', True)
        >>> x, y = generator[0]

    """

    def __init__(
        self,
        inputs: datapack.DataPack,
        task: engine.BaseTask=tasks.Classification(2),
        batch_size: int=32,
        stage: str = 'train',
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
        self.mapping = inputs.mapping
        self.stage = stage
        super().__init__(batch_size, len(inputs.dataframe), shuffle)

    def transform_data(self, inputs: datapack.DataPack) -> dict:
        """Obtain the transformed data from :class:`DataPack`.

        :param inputs: An instance of :class:`DataPack` to be transformed.
        :return: the output of all the transformed inputs.
        """
        data = inputs.dataframe
        out = {}
        for column in data.columns:
            out[column] = np.asarray(data[column])
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
        for key in self.data.keys():
            if key == 'label':
                continue
            batch_x[key] = []
            for val in index_array:
                cont = self.mapping[self.data[key][val]]
                batch_x[key].append(cont)
            batch_x[key] = np.array(batch_x[key])
        batch_x = utils.dotdict(batch_x)
        return (batch_x, batch_y)
