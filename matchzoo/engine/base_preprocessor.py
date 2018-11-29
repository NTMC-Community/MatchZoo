""":class:`BasePreprocessor` define input and ouutput for processors."""

import abc
import typing
from pathlib import Path

import dill

from matchzoo import DataPack
from matchzoo import processor_units


def validate_context(func):
    """Validate context in the preprocessor."""
    def transform_wrapper(self, *args, **kwargs):
        if not self.context:
            raise ValueError(
                'Please fit parameters before transformation.')
        return func(self, *args, **kwargs)

    return transform_wrapper


class BasePreprocessor(metaclass=abc.ABCMeta):
    """:class:`BasePreprocessor` to input handle data."""

    DATA_FILENAME = 'preprocessor.dill'

    def __init__(self):
        """Initialization."""
        self._context = {}

    @property
    def context(self):
        """Return context."""
        return self._context

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
    def transform(self, inputs: list) -> DataPack:
        """
        Transform input data to expected manner.

        This method is an abstract base method, need to be
        implemented in the child class.

        :param inputs: List of text-left, text-right, label triples,
            or list of text-left, text-right tuples.
        """

    def fit_transform(self, inputs: list) -> DataPack:
        """
        Call fit-transform.

        :param inputs: List of text-left, text-right, label triples.
        """
        return self.fit(inputs).transform(inputs)

    def save(self, dirpath: typing.Union[str, Path]):
        """
        Save the :class:`DSSMPreprocessor` object.

        A saved :class:`DSSMPreprocessor` is represented as a directory with
        the `context` object (fitted parameters on training data), it will
        be saved by `pickle`.

        :param dirpath: directory path of the saved :class:`DSSMPreprocessor`.
        """
        dirpath = Path(dirpath)
        data_file_path = dirpath.joinpath(self.DATA_FILENAME)

        if data_file_path.exists():
            raise FileExistsError
        elif not dirpath.exists():
            dirpath.mkdir()

        dill.dump(self, open(data_file_path, mode='wb'))

    @classmethod
    def _default_processor_units(cls) -> list:
        """Prepare needed process units."""
        return [
            processor_units.TokenizeUnit(),
            processor_units.LowercaseUnit(),
            processor_units.PuncRemovalUnit(),
            processor_units.StopRemovalUnit(),
        ]


def load_preprocessor(dirpath: typing.Union[str, Path]) -> DataPack:
    """
    Load the fitted `context`. The reverse function of :meth:`save`.

    :param dirpath: directory path of the saved model.
    :return: a :class:`DSSMPreprocessor` instance.
    """
    dirpath = Path(dirpath)

    data_file_path = dirpath.joinpath(BasePreprocessor.DATA_FILENAME)
    return dill.load(open(data_file_path, 'rb'))
