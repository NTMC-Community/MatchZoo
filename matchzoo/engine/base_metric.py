import typing
import abc


class BaseMetric(abc.ABC):
    ALIAS = 'base_metric'

    @abc.abstractmethod
    def __call__(self, y_true, y_pred):
        """"""

    def __repr__(self):
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
