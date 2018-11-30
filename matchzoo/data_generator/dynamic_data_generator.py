import numpy as np

from matchzoo.data_generator import DataGenerator


class DynamicDataGenerator(DataGenerator):
    """
    Data generator with preprocess unit inside.

    Examples:
        >>> import pandas as pd
        >>> from matchzoo.data_pack import DataPack
        >>> from matchzoo.processor_units import FixedLengthUnit
        >>> relation = [
        ...     ['qid0', 'did0', 1],
        ...     ['qid0', 'did1', 2],
        ...     ['qid0', 'did2', 0]
        ... ]
        >>> left = [['qid0', [1, 2, 3]]]
        >>> right = [
        ...     ['did0', [2, 3, 5, 6]],
        ...     ['did1', [3, 4]],
        ...     ['did2', [4, 5, 7]]
        ... ]
        >>> relation = pd.DataFrame(relation,
        ...                         columns=['id_left', 'id_right', 'label'])
        >>> left = pd.DataFrame(left, columns=['id_left', 'text_left'])
        >>> left.set_index('id_left', inplace=True)
        >>> left['length_left'] = left.apply(lambda x: len(x['text_left']),
        ...                                  axis=1)
        >>> right = pd.DataFrame(right, columns=['id_right', 'text_right'])
        >>> right.set_index('id_right', inplace=True)
        >>> right['length_right'] = right.apply(lambda x: len(x['text_right']),
        ...                                     axis=1)
        >>> input = DataPack(relation=relation,
        ...                  left=left,
        ...                  right=right
        ... )
        >>> fixedlenunit = FixedLengthUnit(3, pad_value=0, pad_mode='post',
        ...     truncate_mode='post')
        >>> data_generator = DynamicDataGenerator(fixedlenunit.transform,
        ...     data_pack=input, batch_size=1, shuffle=False)
        >>> data_generator.num_instance
        3
        >>> len(data_generator)
        3
        >>> x0, y0 = data_generator[0]
        >>> x0['id_left'].tolist()
        ['qid0']
        >>> x0['id_right'].tolist()
        ['did0']
        >>> x0['text_left'].tolist()
        [[1, 2, 3]]
        >>> x0['text_right'].tolist()
        [[2, 3, 5]]
        >>> x0['length_right'].tolist()
        [4]
        >>> y0.tolist()
        [1]
        >>> x1, y1 = data_generator[1]
        >>> x1['id_left'].tolist()
        ['qid0']
        >>> x1['id_right'].tolist()
        ['did1']
        >>> x1['text_right'].tolist()
        [[3, 4, 0]]

    """
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._func = func

    def _get_batch_of_transformed_samples(self, indices: np.array):
        return self._data_pack[indices].apply_on_text(
            self._func, verbose=0).unpack()
