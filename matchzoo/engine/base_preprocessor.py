""":class:`BasePreprocessor` define input and ouutput for processors."""

import abc

from matchzoo import datapack


class BasePreprocessor(metaclass=abc.ABCMeta):
    """:class:`BasePreprocessor` to input handle data."""

    @abc.abstractmethod
    def fit(self) -> 'BasePreprocessor':
        """
        Fit parameters on input data.

        This method is an abstract base method, need to be
        implemented in the child class.

        This method is expected to return itself as a callable
        object.
        """

    @abc.abstractmethod
    def transform(self) -> datapack.DataPack:
        """
        Transform input data to expected manner.

        This method is an abstract base method, need to be
        implemented in the child class.
        """

    def fit_transform(self) -> datapack.DataPack:
        """Call fit-transform."""
        return self.fit().transform()
