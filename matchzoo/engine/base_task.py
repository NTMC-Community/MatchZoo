"""Base task."""

import typing
import abc

from matchzoo.engine import base_metric
from matchzoo.engine import parse_metric


class BaseTask(abc.ABC):
    """Base Task, shouldn't be used directly."""

    def __init__(self, loss=None, metrics=None):
        """
        Base task constructor.

        :param loss: By default the first loss in available losses.
        :param metrics:
        """
        self._loss = loss
        self._metrics = self._convert_metrics(metrics)
        self._assure_loss()
        self._assure_metrics()

    def _convert_metrics(self, metrics):
        if not metrics:
            metrics = []
        elif not isinstance(metrics, list):
            metrics = [metrics]
        return [
            parse_metric.parse_metric(metric, self) for metric in metrics
        ]

    def _assure_loss(self):
        if not self._loss:
            self._loss = self.list_available_losses()[0]

    def _assure_metrics(self):
        if not self._metrics:
            first_available = self.list_available_metrics()[0]
            self._metrics = self._convert_metrics(first_available)

    @property
    def loss(self):
        """:return: Loss used in the task."""
        return self._loss

    @property
    def metrics(self):
        """:return: Metrics used in the task."""
        return self._metrics

    @metrics.setter
    def metrics(
        self,
        new_metrics: typing.Union[
            typing.List[str],
            typing.List[base_metric.BaseMetric],
            str,
            base_metric.BaseMetric
        ]
    ):
        self._metrics = self._convert_metrics(new_metrics)

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
