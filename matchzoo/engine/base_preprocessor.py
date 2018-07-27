""":class:`BasePreprocessor` define input and ouutput for processors."""

import abc

from matchzoo import datapack


class BasePreprocessor(metaclass=abc.ABCMeta):
    """:class:`BasePreprocessor` to input handle data."""

    @abc.abstractmethod
    def fit(self, inputs: list) -> 'BasePreprocessor':
        """
        Fit parameters on input data.

        This method is an abstract base method, need to be
        implemented in the child class.

        This method is expected to return itself as a callable
        object.

        :param inputs: List of text-left, text-right, label triples.
        """

    @abc.abstractmethod
    def transform(self, inputs: list) -> datapack.DataPack:
        """
        Transform input data to expected manner.

        This method is an abstract base method, need to be
        implemented in the child class.

        :param inputs: List of text-left, text-right, label triples,
            or list of text-left, text-right tuples (test stage).
        """

    def fit_transform(self, inputs: list) -> datapack.DataPack:
        """
        Call fit-transform.

        :param inputs: List of text-left, text-right, label triples.
        """
        return self.fit(inputs).transform(inputs)
