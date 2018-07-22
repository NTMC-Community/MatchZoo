"""Matchzoo DataPack, pair-wise tuple (feature) and context as input."""

import typing
from pathlib import Path

import dill
import numpy as np
import pandas as pd


class DataPack(object):
    """
    Matchzoo :class:`DataPack` data structure, store dataframe and context.

    Example:
        >>> features = [([1,3], [2,3]), ([3,0], [1,6])]
        >>> context = {'vocab_size': 2000}
        >>> dp = DataPack(data=features,
        ...               context=context)
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
                 data: typing.Union[list, np.ndarray],
                 context: dict={}):
        """
        Initialize :class:`DataPack`.

        :param data: Input data, could be list-like objects
                     or :class:`numpy.ndarray`.
        :param context: Hyper-parameter fitted during
                        pre-processing stage.
        """
        self._dataframe = pd.DataFrame(data)
        self._context = context

    def __len__(self) -> int:
        """Get numer of rows in the class:`DataPack` object."""
        return self._dataframe.shape[0]

    @property
    def dataframe(self):
        """Get :meth:`dataframe` of :class:`DataPack`."""
        return self._dataframe

    @property
    def context(self):
        """Get :meth:`context` of class:`DataPack`."""
        return self._context

    def sample(self, number, replace=True):
        """
        Sample records from :class:`DataPack` object, for generator.

        :param number: number of records to be sampled, use `batch_size`.
        :param replace: sample with replacement, default value is `True`.

        :return data_pack: return :class:`DataPack` object including
                           sampled data and context (shallow copy of
                           the context`).
        """
        return DataPack(self._dataframe.sample(n=number, replace=replace),
                        self._context.copy())

    def append(self, other: 'DataPack'):
        """
        Append new :class:`DataPack` object to current :class:`DataPack`.

        It should be noted that the context of the previous :class:`DataPack`
        will be updated by the new one.

        :param other: the :class:`DataPack` object to be appended.
        """
        other_dataframe = other.dataframe
        other_context = other.context
        self._dataframe = self._dataframe.append(
            other_dataframe,
            ignore_index=True)
        self.context.update(other_context)

    def save(self, dirpath: typing.Union[str, Path]):
        """
        Save the :class:`DataPack` object.

        A saved :class:`DataPack` is represented as a directory with a
        :class:`DataPack` object (transformed user input as features and
        context), it will be saved by `pickle`.

        :param dirpath: directory path of the saved :class:`DataPack`.
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
    Load a :class:`DataPack`. The reverse function of :meth:`save`.

    :param dirpath: directory path of the saved model.
    :return: a :class:`DataPack` instance.
    """
    dirpath = Path(dirpath)

    data_file_path = dirpath.joinpath(DataPack.DATA_FILENAME)
    dp = dill.load(open(data_file_path, 'rb'))

    return dp
