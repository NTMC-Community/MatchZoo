"""Matchzoo point generator."""

import numpy as np
from matchzoo import engine
from matchzoo import DataPack


class PointGenerator(BaseGenerator):
    """PointGenerator for Matchzoo."""

    def __init__(
            self,
            inputs: DataPack,
            target_mode: str='classification',
            num_class: int=2,
            text_right_max_len: int=5,
            text_left_max_len: int=100,
            is_train: bool=False
        ):
        """Initialization of point-generator."""
        super(PointGenerator, self).__init__(text_left_max_len =
                                             text_left_max_len,
                                             text_right_max_len =
                                             text_right_max_len,
                                             is_train = is_train)
        self.target_mode = target_mode
        self.num_class = num_class
        self.features = inputs.dataframe
        self.context = inputs.context
        self.fill_word = inputs.context['fill_word']
        self.num_samples = len(self.features)

    def get_data() -> tuple:
        """Get all sampels."""
        batch_size = self.num_samples
        t1 = np.zeros((batch_size, self._text_right_max_len), dtype=np.int32)
        t2 = np.zeros((batch_size, self._text_left_max_len), dtype=np.int32)
        t1_len = np.zeros((batch_size,), dtype=np.int32)
        t2_len = np.zeros((batch_size,), dtype=np.int32)
        t1_ids = []
        t2_ids = []
        if self.target_mode == 'regression':
            labels = np.zeros((batch_size,), dtype=np.float32)
        elif self.target_mode == 'classification':
            labels = np.zeros((batch_size, self.num_class), dtype=np.int32)
        else:
            raise ValueError('Error mode for PointGenerator.')

        t1[:] = self.fill_word
        t2[:] = self.fill_word
        for idx, sample in enumerate(self.features.values):
            curr_t1_len = min(self._text_right_max_len, len(sample[0]))
            curr_t2_len = min(self._text_left_max_len, len(sample[1]))
            t1[, :curr_t1_len] = sample[0][:curr_t1_len]
            t2[, :curr_t2_len] = sample[1][:curr_t2_len]
            t1_len[idx] = curr_t1_len
            t2_len[idx] = curr_t2_len
            t1_ids.append(sample[2])
            t2_ids.append(sample[3])
            labels[idx] = sample[4]
        return ({'t1': t1, 't2': t2, 't1_len': t1_len, 't2_len': t2_len,
                     't1_ids': t1_ids, 't2_ids': t2_ids}, labels)

    def get_batch_generator() -> tuple:
        """Get a generator to produce samples dynamically."""
        # to do

