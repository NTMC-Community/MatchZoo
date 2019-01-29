import typing

import matchzoo
from matchzoo.engine.base_metric import BaseMetric
from matchzoo.engine import base_task


def parse_metric(
    metric: typing.Union[str, typing.Type[BaseMetric], BaseMetric],
    task: 'base_task.BaseTask' = None
) -> typing.Union['BaseMetric', str]:
    """
    Parse input metric in any form into a :class:`BaseMetric` instance.

    :param metric: Input metric in any form.
    :param task: Task type for determining specific metric.
    :return: A :class:`BaseMetric` instance

    Examples::
        >>> from matchzoo import metrics
        >>> from matchzoo.engine.parse_metric import parse_metric

    Use `str` as keras native metrics:
        >>> parse_metric('mse')
        'mse'

    Use `str` as MatchZoo metrics:
        >>> mz_metric = parse_metric('map')
        >>> type(mz_metric)
        <class 'matchzoo.metrics.mean_average_precision.MeanAveragePrecision'>

    Use :class:`matchzoo.engine.BaseMetric` subclasses as MatchZoo metrics:
        >>> type(parse_metric(metrics.AveragePrecision))
        <class 'matchzoo.metrics.average_precision.AveragePrecision'>

    Use :class:`matchzoo.engine.BaseMetric` instances as MatchZoo metrics:
        >>> type(parse_metric(metrics.AveragePrecision()))
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
