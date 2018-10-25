import abc


class BaseMetric(abc.ABC):
    ALIAS = 'base_metric'

    @abc.abstractmethod
    def __call__(self, y_true, y_pred):
        """"""

    def __repr__(self):
        return self.ALIAS
