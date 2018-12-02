"""Matchzoo toolkit for token embedding."""

import abc
import csv
import typing

import numpy as np
import pandas as pd


class Embedding(object):
    """
    Embedding class.
    """

    def __init__(self):
        """Init."""
        self._matrix = None

    def _entity_to_vec(self, entity: str):
        """Get word embedding by entity."""
        return self._matrix.loc[entity].values

    def __getitem__(self, entities: typing.Union[str, list]):
        """Get embeddings by entity of list of entities."""
        if isinstance(entities, str):
            return self._entity_to_vec(entities)

        return np.vstack([self._entity_to_vec(entity) for entity in entities])

    @property
    def dimension(self):
        """"""
        return self._matrix.shape[1]

    @property
    def matrix(self):
        """Getter."""
        if not self._matrix:
            raise ValueError('Please load embedding.')
        return self._matrix

    @matrix.setter
    def matrix(self, matrix_instance):
        """Setter."""
        self._matrix = matrix_instance

    @classmethod
    def load(cls, dir: str, embedding_type: str = 'random'):
        """Load embeddings."""
        if embedding_type == 'random':
            self._matrix = pd.DataFrame(np.random.uniform(-random_scale,
                                                          random_scale,
                                                          (len(self._vocab), self._dimension)))
        elif embedding_type == 'word2vec':
            self._matrix = pd.read_table(file_path,
                                         sep=" ",
                                         index_col=0,
                                         header=None,
                                         skiprows=1)
        elif embedding_type == 'glove':
            self._matrix = pd.read_table(file_path,
                                         sep=" ",
                                         index_col=0,
                                         header=None,
                                         quoting=csv.QUOTE_NONE)
        else:
            raise TypeError("Not supported embedding type.")
