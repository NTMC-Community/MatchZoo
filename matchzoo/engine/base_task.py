"""Task types definitions."""

import abc


class BaseTask(abc.ABC):
    """Base Task, shouldn't be used directly."""

    @classmethod
    @abc.abstractmethod
    def list_available_losses(cls) -> list:
        """"""

    @classmethod
    @abc.abstractmethod
    def list_available_metrics(cls) -> list:
        """"""
