import abc


class BaseMetric(abc.ABC):
    ALIAS = 'base_metric'

    @abc.abstractmethod
    def __call__(self, y_true, y_pred):
        """"""

    def __repr__(self):
        return self.ALIAS


def parse_metric(metric):
    if isinstance(metric, BaseMetric):
        return metric
    elif isinstance(metric, str):
        for subclass in BaseMetric.__subclasses__():
            if metric == subclass.ALIAS or metric in subclass.ALIAS:
                return subclass()
        return metric  # keras native metrics
    elif issubclass(metric, BaseMetric):
        return metric()


def compute_metric_list_wise(list_wise_df, metric):
    return list_wise_df.groupby(by='id_left').apply(
            lambda l: metric(l['y_true'], l['y_pred'])).mean()
