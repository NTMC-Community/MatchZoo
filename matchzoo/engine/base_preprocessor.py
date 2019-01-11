""":class:`BasePreprocessor` define input and ouutput for processors."""

import abc
import functools
import typing
from pathlib import Path

import dill

from matchzoo import DataPack
from matchzoo import processor_units


def validate_context(func):
    """Validate context in the preprocessor."""

    @functools.wraps(func)
    def transform_wrapper(self, *args, **kwargs):
        if not self.context:
            raise ValueError('Please call `fit` before calling `transform`.')
        return func(self, *args, **kwargs)

    return transform_wrapper


class BasePreprocessor(metaclass=abc.ABCMeta):
    """
    :class:`BasePreprocessor` to input handle data.

    A preprocessor should be used in two steps. First, `fit`, then,
    `transform`. `fit` collects information into `context`, which includes
    everything the preprocessor needs to `transform` together with other
    useful information for later use. `fit` will only change the
    preprocessor's inner state but not the input data. In contrast,
    `transform` returns a modified copy of the input data without changing
    the preprocessor's inner state.

    """

    DATA_FILENAME = 'preprocessor.dill'

    def __init__(self):
        """Initialization."""
        self._context = {}

    @property
    def context(self):
        """Return context."""
        return self._context

    @abc.abstractmethod
    def fit(self, data_pack: DataPack, verbose=1) -> 'BasePreprocessor':
        """
        Fit parameters on input data.

        This method is an abstract base method, need to be
        implemented in the child class.

        This method is expected to return itself as a callable
        object.

        :param data_pack: :class:`Datapack` object to be fitted.
        :param verbose: Verbosity.
        """

    @abc.abstractmethod
    def transform(self, data_pack: DataPack, verbose=1) -> DataPack:
        """
        Transform input data to expected manner.

        This method is an abstract base method, need to be
        implemented in the child class.

        :param data_pack: :class:`DataPack` object to be transformed.
        :param verbose: Verbosity.
            or list of text-left, text-right tuples.
        """

    def fit_transform(self, data_pack: DataPack, verbose=1) -> DataPack:
        """
        Call fit-transform.

        :param data_pack: :class:`DataPack` object to be processed.
        """
        return self.fit(data_pack, verbose).transform(data_pack, verbose)

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
