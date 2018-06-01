"""Base task."""

import abc


class BaseTask(abc.ABC):
    """Base Task, shouldn't be used directly."""

    @classmethod
    @abc.abstractmethod
    def list_available_losses(cls) -> list:
        """Return a list of available losses."""

    @classmethod
    @abc.abstractmethod
    def list_available_metrics(cls) -> list:
        """Return a list of available metrics."""

    @abc.abstractmethod
    def make_output_layer(self):
        """Return a keras layer to match the task."""


def list_available_tasks(base=BaseTask):
    ret = [base]
    for subclass in base.__subclasses__():
        ret.extend(list_available_tasks(subclass))
    return ret
