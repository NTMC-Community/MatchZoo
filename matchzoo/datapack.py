"""Matchzoo DataPack, pair-wise tuple (feature) and context as input."""

import typing
from pathlib import Path

import dill
import pandas as pd


class DataPack(object):
    """
    Matchzoo DataPack data structure, store dataframe and context.

    Example:
        >>> # features, context generate by processors.
        >>> features = [([1,3], [2,3]), ([3,0], [1,6])]
        >>> context = {'vocab_size': 2000}
        >>> dp = DataPack(data=features,
        ...               context=context)
        >>> # sample without replacement for generation.
        >>> type(dp.sample(1))
        <class 'matchzoo.datapack.DataPack'>
        >>> len(dp)
        2
        >>> features, context = dp.dataframe, dp.context
        >>> context
        {'vocab_size': 2000}
    """

    DATA_FILENAME = 'data.dill'

    def __init__(self,
                 data: list,
                 context: dict={}):
        """Initialize."""
        self._dataframe = pd.DataFrame(data)
        self._context = context

    def __len__(self) -> int:
        """Get size of the data pack."""
        return self._dataframe.shape[0]

    @property
    def dataframe(self):
        """Get data frame."""
        return self._dataframe

    @property
    def context(self):
        """Get context of `DataPack`."""
        return self._context

    def sample(self, number, replace=True):
        """
        Sample records from `DataPack` object, for generator.

        :param number: number of records to be sampled, use `batch_size`.
        :param replace: sample with replacement, default value is `True`.

        :return data_pack: return `DataPack` object including sampled data
                           and context.
        """
        return DataPack(self._dataframe.sample(n=number, replace=replace),
                        self._context)

    def append(self, new_data_pack: 'DataPack', overwite_context: bool=True):
        """
        Append a new `DataPack` object to current `DataPack` object.

        It should be noted that the context of the previous `DataPack`
        will be updated by the new one.

        :param new_data_pack: A new DataPack object.
        :param overwite_context: Allow overwrite common context by new context.
        """
        new_dataframe = new_data_pack.dataframe
        new_context = new_data_pack.context
        self._dataframe = self._dataframe.append(
            new_dataframe,
            ignore_index=True)
        if overwite_context:
            self._context = {**self._context, **new_context}
        else:
            self._context = {**new_context, **self._context}

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

        if dirpath.exists():
            raise FileExistsError
        else:
            dirpath.mkdir()

        data_file_path = dirpath.joinpath(self.DATA_FILENAME)
        dill.dump(self, open(data_file_path, mode='wb'))


def load_datapack(dirpath: typing.Union[str, Path]) -> DataPack:
    """
    Load a `DataPack`. The reverse function of :meth:`DataPack.save`.

    :param dirpath: directory path of the saved model
    :return: a :class:`DataPack` instance
    """
    dirpath = Path(dirpath)

    data_file_path = dirpath.joinpath(DataPack.DATA_FILENAME)
    dp = dill.load(open(data_file_path, 'rb'))

    return dp