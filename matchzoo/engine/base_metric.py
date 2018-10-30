"""Metric base class and some related utilities."""

import typing
import abc

import numpy as np


class BaseMetric(abc.ABC):
    """Metric base class."""

    ALIAS = 'base_metric'

    @abc.abstractmethod
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Call to compute the metric.

        :param y_true: An array of groud truth labels.
        :param y_pred: An array of predicted values.
        :return: Evaluation of the metric.
        """

    def __repr__(self):
        """:return: Formated string representation of the metric."""
        return self.ALIAS


def parse_metric(metric: typing.Union[str, BaseMetric]):
    """
    Parse input metric in any form into a :class:`BaseMetric` instance.

    :param metric: Input metric in any form.
    :return: A :class:`BaseMetric` instance
    """
    if isinstance(metric, BaseMetric):
        return metric
    elif isinstance(metric, str):
        metric = metric.lower()  # ignore case
        for subclass in BaseMetric.__subclasses__():
            if metric == subclass.ALIAS or metric in subclass.ALIAS:
                return subclass()
        return metric  # keras native metrics
    elif issubclass(metric, BaseMetric):
        return metric()
