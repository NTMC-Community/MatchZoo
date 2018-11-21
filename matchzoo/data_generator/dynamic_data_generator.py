import numpy as np

from .data_generator import DataGenerator


class DynamicDataGenerator(DataGenerator):
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._func = func

    def _get_batch_of_transformed_samples(self, indices: np.array):
        return self._data_pack[indices].apply_on_text(self._func).unpack()
