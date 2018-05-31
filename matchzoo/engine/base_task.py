"""Task types definitions."""

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
