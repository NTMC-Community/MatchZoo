"""Base task."""

import typing
import abc

import keras


class BaseTask(abc.ABC):
    """Base Task, shouldn't be used directly."""

    @classmethod
    @abc.abstractmethod
    def list_available_losses(cls) -> list:
        """:return: a list of available losses."""

    @classmethod
    @abc.abstractmethod
    def list_available_metrics(cls) -> list:
        """:return: a list of available metrics."""

    @abc.abstractmethod
    def make_output_layer(self) -> keras.layers.Dense:
        """:return: a keras layer to match the task."""


def list_available_tasks(base=BaseTask) -> typing.List[typing.Type[BaseTask]]:
    """:return: a list of available task types."""
    ret = [base]
    for subclass in base.__subclasses__():
        ret.extend(list_available_tasks(subclass))
    return ret
