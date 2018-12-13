"""Dynamic data generator with transform function inside."""
import numpy as np

from matchzoo.data_generator import DataGenerator


class DynamicDataGenerator(DataGenerator):
    """
    Data generator with preprocess unit inside.

    Examples:
        >>> import matchzoo as mz
        >>> input = mz.datasets.toy.load_data()
        >>> data_generator = DynamicDataGenerator(len,
        ...     data_pack=input, batch_size=1, shuffle=False)
        >>> data_generator.num_instance
        49
        >>> len(data_generator)
        49
        >>> x0, y0 = data_generator[0]
        >>> x0['id_left'].tolist()
        ['q1']
        >>> x0['id_right'].tolist()
        ['d1']
        >>> x0['text_left'].tolist()
        [73]
        >>> x0['text_right'].tolist()
        [59]
        >>> y0.tolist()
        [[0.0]]
        >>> x1, y1 = data_generator[1]
        >>> x1['id_left'].tolist()
        ['q2']
        >>> x1['id_right'].tolist()
        ['d2']
        >>> x1['text_left'].tolist()
        [30]
        >>> x1['text_right'].tolist()
        [41]
        >>> y1.tolist()
        [[1.0]]

    """

    def __init__(self, func, *args, **kwargs):
        """:class:`DynamicDataGenerator` constructor."""
        super().__init__(*args, **kwargs)
        self._func = func

    def _get_batch_of_transformed_samples(self, indices: np.array):
        """
        Get a batch of samples based on their ids.

        :param indices: A list of instance ids.
        :return: A batch of transformed samples.
        """
        return self._data_pack[indices].apply_on_text(
            self._func, verbose=0).unpack()
