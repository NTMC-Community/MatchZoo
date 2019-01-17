"""Base task."""

import typing
import abc

from matchzoo import engine


class BaseTask(abc.ABC):
    """Base Task, shouldn't be used directly."""

    @classmethod
    def convert_metrics(cls, metrics: typing.Union[list,
                                                   str,
                                                   engine.BaseMetric]
                        ) -> typing.List[engine.BaseMetric]:
        """
        Convert `metrics` into properly formed list of metrics.

        Examples:
            >>> BaseTask.convert_metrics(['mse'])
            ['mse']
            >>> BaseTask.convert_metrics('map')
            [mean_average_precision(0.0)]

        """
        if not metrics:
            metrics = []
        elif not isinstance(metrics, list):
            metrics = [metrics]
        return [engine.parse_metric(metric) for metric in metrics]

    def __init__(self, loss=None, metrics=None):
        """
        Base task constructor.

        :param loss: By default the first loss in available losses.
        :param metrics:
        """
        self._loss = loss
        self._metrics = metrics

        self._assure_loss()
        self._assure_metrics()

        self._metrics = self.convert_metrics(self._metrics)

    def _assure_loss(self):
        if not self._loss:
            self._loss = self.list_available_losses()[0]

    def _assure_metrics(self):
        if not self._metrics:
            self._metrics = [self.list_available_metrics()[0]]

    @property
    def loss(self):
        """:return: Loss used in the task."""
        return self._loss

    @property
    def metrics(self):
        """:return: Metrics used in the task."""
        return self._metrics

    @metrics.setter
    def metrics(self, new_metrics: typing.Union[list, str, engine.BaseMetric]):
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
