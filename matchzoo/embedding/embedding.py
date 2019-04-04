"""Matchzoo toolkit for token embedding."""

import csv
import typing

import numpy as np
import pandas as pd

import matchzoo as mz


class Embedding(object):
    """
    Embedding class.

    Examples::
        >>> import matchzoo as mz
        >>> train_raw = mz.datasets.toy.load_data()
        >>> pp = mz.preprocessors.NaivePreprocessor()
        >>> train = pp.fit_transform(train_raw, verbose=0)
        >>> vocab_unit = mz.build_vocab_unit(train, verbose=0)
        >>> term_index = vocab_unit.state['term_index']
        >>> embed_path = mz.datasets.embeddings.EMBED_RANK

    To load from a file:
        >>> embedding = mz.embedding.load_from_file(embed_path)
        >>> matrix = embedding.build_matrix(term_index)
        >>> matrix.shape[0] == len(term_index) + 1
        True

    To build your own:
        >>> data = pd.DataFrame(data=[[0, 1], [2, 3]], index=['A', 'B'])
        >>> embedding = mz.Embedding(data)
        >>> matrix = embedding.build_matrix({'A': 2, 'B': 1})
        >>> matrix.shape == (3, 2)
        True

    """

    def __init__(self, data: pd.DataFrame):
        """
        Embedding.

        :param data: DataFrame to use as term to vector mapping.
        """
        self._data = data

    @property
    def input_dim(self) -> int:
        """:return Embedding input dimension."""
        return self._data.shape[0]

    @property
    def output_dim(self) -> int:
        """:return Embedding output dimension."""
        return self._data.shape[1]

    def build_matrix(
        self,
        term_index: typing.Union[
            dict, mz.preprocessors.units.Vocabulary.TermIndex],
        initializer=lambda: np.random.uniform(-0.2, 0.2)
    ) -> np.ndarray:
        """
        Build a matrix using `term_index`.

        :param term_index: A `dict` or `TermIndex` to build with.
        :param initializer: A callable that returns a default value for missing
            terms in data. (default: a random uniform distribution in range)
            `(-0.2, 0.2)`).
        :return: A matrix.
        """
        input_dim = len(term_index) + 1

        matrix = np.empty((input_dim, self.output_dim))
        for index in np.ndindex(*matrix.shape):
            matrix[index] = initializer()

        valid_keys = set(self._data.index)
        for term, index in term_index.items():
            if term in valid_keys:
                matrix[index] = self._data.loc[term]

        return matrix


def load_from_file(file_path: str, mode: str = 'word2vec') -> Embedding:
    """
    Load embedding from `file_path`.

    :param file_path: Path to file.
    :param mode: Embedding file format mode, one of 'word2vec' or 'glove'.
        (default: 'word2vec')
    :return: An :class:`matchzoo.embedding.Embedding` instance.
    """
    if mode == 'word2vec':
        data = pd.read_csv(file_path,
                           sep=" ",
                           index_col=0,
                           header=None,
                           skiprows=1)
    elif mode == 'glove':
        data = pd.read_csv(file_path,
                           sep=" ",
                           index_col=0,
                           header=None,
                           quoting=csv.QUOTE_NONE)
    else:
        raise TypeError(f"{mode} is not a supported embedding type."
                        f"`word2vec` or `glove` expected.")
    return Embedding(data)
