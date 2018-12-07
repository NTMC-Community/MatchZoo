"""Matchzoo toolkit for token embedding."""

import csv
import typing

import numpy as np
import pandas as pd

from matchzoo import processor_units


class Embedding(object):
    """
    Embedding class.

    Examples::
        >>> import matchzoo as mz
        >>> data_pack = mz.datasets.toy.load_train_rank_data()
        >>> pp = mz.preprocessors.NaivePreprocessor()
        >>> vocab_unit = mz.build_vocab(pp.fit_transform(data_pack))
        >>> term_index = vocab_unit.state['term_index']
        >>> embed_path = mz.datasets.embeddings.EMBED_RANK

    To load from a file:
        >>> embedding = mz.embedding.load_from_file(embed_path)
        >>> matrix = embedding.build_matrix(term_index)
        >>> matrix.shape[0] == len(term_index) + 1
        True

    To build your own:
        >>> data = pd.read_table(embed_path, sep=" ", index_col=0,
        ...                      header=None, skiprows=1)
        >>> embedding = mz.embedding.Embedding(data)
        >>> matrix = embedding.build_matrix(term_index)
        >>> matrix.shape[0] == len(term_index) + 1
        True

    """

    def __init__(self, data: pd.DataFrame):
        """
        Embedding.

        :param data: DataFrame to use as term to vector mapping.
        """
        self._data = data

    @property
    def output_dim(self) -> int:
        """:return Embedding output dimension."""
        return self._data.shape[1]

    def build_matrix(
        self,
        term_index: typing.Union[
            dict, processor_units.VocabularyUnit.TermIndex],
        initializer=lambda shape: np.random.uniform(-0.2, 0.2, shape)
    ) -> np.ndarray:
        """
        Build a matrix using `term_index`.

        :param term_index: A `dict` or `TermIndex` to build with.
        :param initializer: Initializer to initialize missing terms in data.
            Should takes `shape` as arguments and returns a initialized matrix.
            (default: a random uniform distribution in `(-0.2, 0.2)`)
        :return: A matrix.
        """
        input_dim = len(term_index) + 1
        matrix = initializer((input_dim, self.output_dim))
        for term, index in term_index.items():
            if term in self._data.index:
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
        matrix = pd.read_table(file_path,
                               sep=" ",
                               index_col=0,
                               header=None,
                               skiprows=1)
    elif mode == 'glove':
        matrix = pd.read_table(file_path,
                               sep=" ",
                               index_col=0,
                               header=None,
                               quoting=csv.QUOTE_NONE)
    else:
        raise TypeError("Not supported embedding type.")
    return Embedding(matrix)
