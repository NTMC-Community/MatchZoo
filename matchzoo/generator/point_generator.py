"""Matchzoo point generator."""

import numpy as np
from matchzoo import engine


class PointGenerator(BaseGenerator):
    """PointGenerator for Matchzoo."""

    def __init__(
            self,
            inputs: DataPack,
            target_mode: str='classification',
            num_class: int=2,
            text_right_max_len: int=5,
            text_left_max_len: int=100,
            batch_size: int=32,
            shuffle: bool=True,
            is_train: bool=False
        ):
        """Initialization of point-generator."""
        self._text_left_max_len = text_left_max_len
        self._text_right_max_len = text_right_max_len
        self.target_mode = target_mode
        self.num_class = num_class
        self.transform_data(inputs)
        super(PointGenerator, self).__init__(
                                             batch_size = batch_size,
                                             shuffle = shuffle,
                                             is_train = is_train
                                             )

    def transform_data(inputs: DataPack):
        """Obtain the transformed data from datapack."""
        data = inputs.dataframe
        context = inputs.context
        fill_word = context['fill_word']
        text_left = []
        text_right = []
        labels = []
        ids = []
        for idx, sample in enumerate(data.values):
            x0 = sample[0]
            x1 = sample[1]
            y = sample[2]
            x0_id = sample[3]
            x1_id = sample[4]
            ctl_len = min(self.text_left_max_len, len(x0))
            ctr_len = min(self.text_right_max_len, len(x1))
            xx0 = np.zeros((self.text_left_max_len,), dtype=np.int32)
            xx1 = np.zeros((self.text_right_max_len,), dtype=np.int32)
            xx0[:] = self.fill_word # padding
            xx1[:] = self.fill_word # padding
            xx0[:ctl] = x0[:ctl]
            xx1[:ctr] = x1[:ctr]
            text_left.append(xx0)
            text_right.append(xx1)
            labels.append(y)
            ids.append((x0_id, x1_id))
        self.x_left = np.asarray(text_left)
        self.x_right = np.asarray(text_right)
        self.y = np.asarray(labels)
        self.ids = np.asarray(ids)

    def _total_num_instances(self):
        return len(self.x_left)

    def _get_batches_of_transformed_samples(self, index_array):
        """Get all sampels."""
        batch_size = len(index_array)
        batch_x_left = np.zeros((batch_size, self.text_left_max_len),
                                dtype=np.int32)
        batch_x_right = np.zeros((batch_size, self.text_right_max_len),
                                 dtype=np.int32)
        batch_ids = []
        if self.target_mode == 'regression':
            batch_y = np.zeros((batch_size,), dtype=np.float32)
            batch_y[:] = self.y[index_array]
        elif self.target_mode == 'classification':
            batch_y = np.zeros((batch_size, self.num_class))
            for i, label in enumerate(self.y[index_array]):
                batch_y[i, label] = 1
        else:
            raise ValueError('Error target mode in point generator.')
        for i, j in enumerate(index_array):
            batch_x_left[i] = self.x_left[j]
            batch_x_right[i] = self.x_right[j]
            batch_ids.append(self.ids[j])

        return ({'x_left': batch_x_left, 'x_right': batch_x_right, 'ids':
                 batch_ids}, batch_y)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch
        """
        # Keep under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of data is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_sample(index_array)

