"""Precision for ranking."""
import numpy as np

from matchzoo.engine.base_metric import BaseMetric, sort_and_couple


class Precision(BaseMetric):
    """Precision metric."""

    ALIAS = 'precision'

    def __init__(self, k: int = 1, threshold: float = 0.):
        """
        :class:`PrecisionMetric` constructor.

        :param k: Number of results to consider.
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate precision@k.

        Example:
            >>> y_true = [0, 0, 0, 1]
            >>> y_pred = [0.2, 0.4, 0.3, 0.1]
            >>> Precision(k=1)(y_true, y_pred)
            0.0
            >>> Precision(k=2)(y_true, y_pred)
            0.0
            >>> Precision(k=4)(y_true, y_pred)
            0.25
            >>> Precision(k=5)(y_true, y_pred)
            0.2

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Precision @ k
        :raises: ValueError: len(r) must be >= k.
        """
        if self._k <= 0:
            raise ValueError(f"k must be greater than 0."
                             f"{self._k} received.")
        coupled_pair = sort_and_couple(y_true, y_pred)
        precision = 0.0
        for idx, (label, score) in enumerate(coupled_pair):
            if idx >= self._k:
                break
            if label > self._threshold:
                precision += 1.
        return precision / self._k
