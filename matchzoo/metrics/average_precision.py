"""Average precision metric for ranking."""
import numpy as np

from matchzoo import engine
from . import Precision


class AveragePrecision(engine.BaseMetric):
    """Average precision metric."""

    ALIAS = ['average_precision', 'ap']

    def __init__(self, threshold=0):
        """
        :class:`AveragePrecision` constructor.

        :param threshold: The label threshold of relevance degree.
        """
        self._threshold = threshold

    def __repr__(self):
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}({self._threshold})"

    def __call__(self, y_true, y_pred):
        """
        Calculate average precision (area under PR curve).

        Example:
            >>> y_true = [0, 1]
            >>> y_pred = [0.1, 0.6]
            >>> round(AveragePrecision()(y_true, y_pred), 2)
            0.75
            >>> round(AveragePrecision()([], []), 2)
            0.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Average precision.
        """
        precision_metrics = [Precision(k + 1) for k in range(len(y_pred))]
        out = [metric(y_true, y_pred) for metric in precision_metrics]
        if not out:
            return 0.
        return np.mean(out)
