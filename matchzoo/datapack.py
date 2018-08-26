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
        >>> content = {
        ...     'qid1':'query 1',
        ...     'qid2':'query 2',
        ...     'did1':'document 1',
        ...     'did2':'document 2'
        ... }
        >>> relation = [['qid1', 'did1', 1], ['qid2', 'did2', 1]]
        >>> context = {'vocab_size': 2000}
        >>> dp = DataPack(
        ...     relation=relation,
        ...     content=content,
        ...     context=context
        ... )
        >>> len(dp)
        2
        >>> relation, context = dp.relation, dp.context
        >>> context
        {'vocab_size': 2000}
    """

    DATA_FILENAME = 'data.dill'

    def __init__(self,
                 relation: typing.Union[list, np.ndarray],
                 content: dict,
                 context: dict={},
                 columns: list=None):
        """
        Initialize :class:`DataPack`.

        :param relation: Store the relation between left document
            and right document use ids.
        :param content: Store the content of ids.
        :param context: Hyper-parameter fitted during
            pre-processing stage.
        :param columns: List of column names of the :attr:`relation`
            variable.
        """
        self._relation = pd.DataFrame(relation, columns=columns)
        self._content = content
        self._context = context

    def __len__(self) -> int:
        """Get numer of rows in the class:`DataPack` object."""
        return self._relation.shape[0]

    @property
    def relation(self):
        """Get :meth:`relation` of :class:`DataPack`."""
        return self._relation

    @property
    def content(self):
        """Get :meth:`content` of :class:`DataPack`."""
        return self._content

    @content.setter
    def content(self, value: dict):
        """Set the value of :attr:`content`."""
        self._content = value

    @property
    def context(self):
        """Get :meth:`context` of class:`DataPack`."""
        return self._context

    @context.setter
    def context(self, value: dict):
        """Set the value of :attr:`context`."""
        self._context = value

    def save(self, dirpath: typing.Union[str, Path]):
        """
        Save the :class:`DataPack` object.

        A saved :class:`DataPack` is represented as a directory with a
        :class:`DataPack` object (transformed user input as features and
        context), it will be saved by `pickle`.

        :param dirpath: directory path of the saved :class:`DataPack`.
        """
        dirpath = Path(dirpath)
        data_file_path = dirpath.joinpath(self.DATA_FILENAME)

        if data_file_path.exists():
            raise FileExistsError
        elif not dirpath.exists():
            dirpath.mkdir()

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
