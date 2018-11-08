"""Matchzoo DataPack, pair-wise tuple (feature) and context as input."""

import typing
from pathlib import Path

import dill
import pandas as pd


class DataPack(object):
    """
    Matchzoo :class:`DataPack` data structure, store dataframe and context.

    Example:
        >>> left = [
        ...     ['qid1', 'query 1', 'feature 1'],
        ...     ['qid2', 'query 2', 'feature 2']
        ... ]
        >>> right = [
        ...     ['did1', 'document 1'],
        ...     ['did2', 'document 2']
        ... ]
        >>> relation = [['qid1', 'did1', 1], ['qid2', 'did2', 1]]
        >>> context = {'vocab_size': 2000}
        >>> relation_df = pd.DataFrame(relation)
        >>> left = pd.DataFrame(left)
        >>> right = pd.DataFrame(right)
        >>> dp = DataPack(
        ...     relation=relation_df,
        ...     left=left,
        ...     right=right,
        ... )
        >>> len(dp)
        2
        >>> relation, context = dp.relation, dp.context
        >>> context['vocab_size']
        2000
    """

    DATA_FILENAME = 'data.dill'

    def __init__(
        self,
        relation: pd.DataFrame,
        left: pd.DataFrame,
        right: pd.DataFrame
    ):
        """
        Initialize :class:`DataPack`.

        :param relation: Store the relation between left document
            and right document use ids.
        :param left: Store the content or features for id_left.
        :param right: Store the content or features for
            id_right.
        """
        self._relation = relation
        self._left = left
        self._right = right

    @property
    def stage(self):
        if 'label' in self._relation.columns:
            return 'train'
        else:
            return 'predict'

    def __len__(self) -> int:
        """Get numer of rows in the class:`DataPack` object."""
        return self._relation.shape[0]

    @property
    def relation(self) -> pd.DataFrame:
        """Get :meth:`relation` of :class:`DataPack`."""
        return self._relation

    @property
    def left(self) -> pd.DataFrame:
        """Get :meth:`left` of :class:`DataPack`."""
        return self._left

    @left.setter
    def left(self, value: pd.DataFrame):
        """Set the value of :attr:`left`.

        Note the value should be indexed with column name.
        """
        self._left = value

    @property
    def right(self) -> pd.DataFrame:
        """Get :meth:`right` of :class:`DataPack`."""
        return self._right

    @right.setter
    def right(self, value: pd.DataFrame):
        """Set the value of :attr:`right`.

        Note the value should be indexed with column name.
        """
        self._right = value

    def copy(self):
        return DataPack(left=self._left.copy(),
                        right=self._right.copy(),
                        relation=self._relation.copy())

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
