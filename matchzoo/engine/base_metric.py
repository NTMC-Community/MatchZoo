"""Metric base class and some related utilities."""

import typing
import abc

import numpy as np

import matchzoo


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

    @abc.abstractmethod
    def __repr__(self):
        """:return: Formated string representation of the metric."""

    def __eq__(self, other):
        """:return: `True` if two metrics are equal, `False` otherwise."""
        return (type(self) is type(other)) and (vars(self) == vars(other))

    def __hash__(self):
        """:return: Hashing value using the metric as `str`."""
        return str(self).__hash__()


def sort_and_couple(labels: np.array, scores: np.array) -> np.array:
    """Zip the `labels` with `scores` into a single list."""
    couple = list(zip(labels, scores))
    return np.array(sorted(couple, key=lambda x: x[1], reverse=True))


def parse_metric(
    metric: typing.Union[str, typing.Type[BaseMetric], BaseMetric],
    task: 'matchzoo.engine.BaseTask' = None
) -> typing.Union['BaseMetric', str]:
    """
    Parse input metric in any form into a :class:`BaseMetric` instance.

    :param metric: Input metric in any form.
    :param task: Task type for determining specific metric.
    :return: A :class:`BaseMetric` instance

    Examples::
        >>> from matchzoo import engine, metrics

    Use `str` as keras native metrics:
        >>> engine.parse_metric('mse')
        'mse'

    Use `str` as MatchZoo metrics:
        >>> mz_metric = engine.parse_metric('map')
        >>> type(mz_metric)
        <class 'matchzoo.metrics.mean_average_precision.MeanAveragePrecision'>

    Use :class:`matchzoo.engine.BaseMetric` subclasses as MatchZoo metrics:
        >>> type(engine.parse_metric(metrics.AveragePrecision))
        <class 'matchzoo.metrics.average_precision.AveragePrecision'>

    Use :class:`matchzoo.engine.BaseMetric` instances as MatchZoo metrics:
        >>> type(engine.parse_metric(metrics.AveragePrecision()))
        <class 'matchzoo.metrics.average_precision.AveragePrecision'>

    """
    if task is None:
        task = matchzoo.tasks.Ranking()

    if isinstance(metric, str):
        metric = metric.lower()  # ignore case

        # matchzoo metrics in str form
        for subclass in BaseMetric.__subclasses__():
            if metric == subclass.ALIAS or metric in subclass.ALIAS:
                return subclass()

        # keras native metrics
        return _remap_keras_metric(metric, task)
    elif isinstance(metric, BaseMetric):
        return metric
    elif issubclass(metric, BaseMetric):
        return metric()
    else:
        raise ValueError(metric)


def _remap_keras_metric(metric: str, task) -> str:
    # we do not support sparse label in classification.
    lookup = {
        matchzoo.tasks.Ranking: {
            'acc': 'binary_accuracy',
            'accuracy': 'binary_accuracy',
            'crossentropy': 'binary_crossentropy',
            'ce': 'binary_crossentropy',
        },
        matchzoo.tasks.Classification: {
            'acc': 'categorical_accuracy',
            'accuracy': 'categorical_accuracy',
            'crossentropy': 'categorical_crossentropy',
            'ce': 'categorical_crossentropy',
        }
    }
    return lookup[type(task)].get(metric, metric)
