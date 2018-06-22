"""Matchzoo DataPack, pair-wise tuple (feature) and context as input."""

import typing
from pathlib import Path

import dill
import pandas as pd


class DataPack(pd.DataFrame):
    """
    Matchzoo DataPack data structure, store dataframe and context.

    Example:
        >>> # features, context generate by processors.
        >>> features = [([1,3], [2,3]), ([3,0], [1,6])]
        >>> context = {'vocab_size': 2000}
        >>> dp = DataPack(data=features,
        ...               context=context)
        >>> dp.context
        {'vocab_size': 2000}
        >>> # sample without replacement for generation.
        >>> type(dp.sample(1))
        <class 'matchzoo.datapack.DataPack'>
        >>> dp.size
        2
        >>> features, context = dp.unpack()
    """

    _metadata = ['context']

    DATA_FILENAME = 'data.pkl'
    CONTEXT_FILENAME = 'context.pkl'

    def __init__(self,
                 data: list,
                 context: dict={},
                 index: list= None,
                 columns: list=['text_left', 'text_right'],
                 dtype: object=None,
                 copy: bool=True):
        """Initialize."""
        super(DataPack, self).__init__(data=data,
                                       index=index,
                                       columns=columns,
                                       dtype=dtype,
                                       copy=copy)
        if self.shape[1] != 2:
            raise ValueError(
                "Pair-wise input expected.")
        self.context = context

    @property
    def _constructor(self) -> callable:
        """Subclass pd.DataFrame."""
        return DataPack._internal_ctor

    @classmethod
    def _internal_ctor(cls, *args, **kwargs):
        """Create subclass inputs to store context."""
        kwargs['context'] = None
        return cls(*args, **kwargs)

    @property
    def size(self) -> int:
        """Get size of the data pack."""
        return self.shape[0]

    def unpack(self) -> typing.Union[pd.DataFrame, dict]:
        """Unpack DataPack."""
        return self, self.context

    def save(self, dirpath: typing.Union[str, Path]):
        """
        Save the `DataPack` object.

        A saved `DataPack` is represented as a directory with two files.
        One is a `DataPack` records (transformed user input as features),
        the otehr one is fitted context parameters such as `vocab_size`.
        Both of them will be saved by `pickle`.

        :param dirpath: directory path of the saved `DataPack`.
        """
        dirpath = Path(dirpath)
        print(dirpath)

        if dirpath.exists():
            raise FileExistsError
        else:
            dirpath.mkdir()

        data_file_path = dirpath.joinpath(self.DATA_FILENAME)
        dill.dump(self, open(data_file_path, mode='wb'))

        context_file_path = dirpath.joinpath(self.CONTEXT_FILENAME)
        dill.dump(self.context, open(context_file_path, mode='wb'))


def load_datapack(dirpath: typing.Union[str, Path]) -> DataPack:
    """
    Load a `DataPack`. The reverse function of :meth:`DataPack.save`.

    :param dirpath: directory path of the saved model
    :return: a :class:`DataPack` instance
    """
    dirpath = Path(dirpath)

    data_file_path = dirpath.joinpath(DataPack.DATA_FILENAME)
    data = dill.load(open(data_file_path, 'rb'))

    context_file_path = dirpath.joinpath(DataPack.CONTEXT_FILENAME)
    context = dill.load(open(context_file_path, 'rb'))

    return DataPack(data=data, context=context)
