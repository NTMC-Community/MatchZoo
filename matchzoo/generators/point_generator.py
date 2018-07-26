"""Matchzoo point generator."""

import numpy as np
from matchzoo import engine
from matchzoo import tasks
from matchzoo.datapack import DataPack


class PointGenerator(engine.BaseGenerator):
    """PointGenerator for Matchzoo.

    Ponit generator can be used for classification as well as ranking.

    # Arguments
        inputs: the transformed dataset
    """

    def __init__(
        self,
        inputs: DataPack,
        task_type: engine.BaseTask,
        batch_size: int=32,
        shuffle: bool=True
    ):
        """Initialization of point-generator."""
        self.task_type = task_type
        self.n = len(inputs)
        self.transform_data(inputs)
        super(PointGenerator, self).__init__(batch_size, shuffle)

    def transform_data(self, inputs: DataPack):
        """Obtain the transformed data from datapack."""
        data = inputs.dataframe
        self.x_left = np.asarray(data.text_left)
        self.x_right = np.asarray(data.text_right)
        self.y = np.asarray(data.label)
        self.ids = np.asarray(data.id)

    def _get_batches_of_transformed_samples(self, index_array):
        """Get all sampels."""
        batch_size = len(index_array)
        batch_x_left = []
        batch_x_right = []
        batch_ids = []
        if self.task_type == tasks.Regression:
            batch_y = self.y
        elif self.target_mode == tasks.Classification:
            batch_y = np.zeros((batch_size, self.task_type._num_classes),
                               dtype=np.int32)
            for i, label in enumerate(self.y[index_array]):
                batch_y[i, label] = 1
        else:
            raise ValueError('Error target mode in point generator.')

        for i, j in enumerate(index_array):
            batch_x_left[i] = self.x_left[j]
            batch_x_right[i] = self.x_right[j]
            batch_ids.append(self.ids[j])

        return ({'x_left': np.ndarray(batch_x_left), 'x_right':
                 np.ndarray(batch_x_right), 'ids':
                 batch_ids}, np.ndarray(batch_y))
