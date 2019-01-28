import abc
import typing


class Unit(metaclass=abc.ABCMeta):
    """Process unit do not persive state (i.e. do not need fit)."""

    @abc.abstractmethod
    def transform(self, input_: typing.Any):
        """Abstract base method, need to be implemented in subclass."""
