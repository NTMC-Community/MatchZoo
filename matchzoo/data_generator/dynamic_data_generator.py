"""Dynamic data generator with transform function inside."""
import numpy as np

from matchzoo.data_generator import DataGenerator


class DynamicDataGenerator(DataGenerator):
    """
    Data generator with preprocess unit inside.

    Examples:
        >>> import matchzoo as mz
        >>> raw_data = mz.datasets.toy.load_data()
        >>> data_generator = DynamicDataGenerator(len, data_pack=raw_data,
        ...                                       batch_size=1, shuffle=False)
        >>> len(data_generator)
        100
        >>> x, y = data_generator[0]
        >>> type(x)
        <class 'dict'>
        >>> x.keys()
        dict_keys(['id_left', 'text_left', 'id_right', 'text_right'])
        >>> type(x['id_left'])
        <class 'numpy.ndarray'>
        >>> type(x['id_right'])
        <class 'numpy.ndarray'>
        >>> type(x['text_left'])
        <class 'numpy.ndarray'>
        >>> type(x['text_right'])
        <class 'numpy.ndarray'>
        >>> type(y)
        <class 'numpy.ndarray'>

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
