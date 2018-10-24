import abc


class BaseMetric(abc.ABC):
    @abc.abstractmethod
    def __call__(self, y_true, y_pred):
        """"""
