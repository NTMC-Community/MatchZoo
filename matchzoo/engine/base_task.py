"""Base task."""

import typing
import abc


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

    @property
    @abc.abstractmethod
    def output_shape(self) -> tuple:
        """:return: output shape of a single sample of the task."""


def list_available_tasks(base=BaseTask) -> typing.List[typing.Type[BaseTask]]:
    """:return: a list of available task types."""
    ret = [base]
    for subclass in base.__subclasses__():
        ret.extend(list_available_tasks(subclass))
    return ret
