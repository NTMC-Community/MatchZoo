"""Base task."""

import typing
import abc

from matchzoo import engine


class BaseTask(abc.ABC):
    """Base Task, shouldn't be used directly."""

    @classmethod
    def convert_metrics(cls, metrics):
        if not metrics:
            metrics = []
        elif not isinstance(metrics, list):
            metrics = [metrics]
        for i, metric in enumerate(metrics):
            if issubclass(metric, engine.BaseMetric):
                metrics[i] = metric()
        return metrics

    def __init__(self, loss=None, metrics=None):
        self._loss = loss

        self.assure_loss()
        self.assure_metrics()

        self._metrics = self.convert_metrics(metrics)

    def assure_loss(self):
        if not self._loss:
            self._loss = self.list_available_losses()[0]

    def assure_metrics(self):
        if not self._metrics:
            self._metrics = [self.list_available_metrics()[0]]

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, new_metrics):
        self._metrics = self.convert_metrics(new_metrics)

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

    @property
    @abc.abstractmethod
    def output_dtype(self):
        """:return: output data type for specific task."""


def list_available_tasks(base=BaseTask) -> typing.List[typing.Type[BaseTask]]:
    """:return: a list of available task types."""
    ret = [base]
    for subclass in base.__subclasses__():
        ret.extend(list_available_tasks(subclass))
    return ret
