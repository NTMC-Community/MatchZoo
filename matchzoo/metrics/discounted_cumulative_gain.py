"""Discounted cumulative gain metric for ranking."""
import math

import numpy as np

from matchzoo.engine.base_metric import BaseMetric, sort_and_couple


class DiscountedCumulativeGain(BaseMetric):
    """Disconunted cumulative gain metric."""

    ALIAS = ['discounted_cumulative_gain', 'dcg']

    def __init__(self, k: int = 1, threshold: float = 0.):
        """
        :class:`DiscountedCumulativeGain` constructor.

        :param k: Number of results to consider.
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate discounted cumulative gain (dcg).

        Relevance is positive real values or binary values.

        Example:
            >>> y_true = [0, 1, 2, 0]
            >>> y_pred = [0.4, 0.2, 0.5, 0.7]
            >>> DiscountedCumulativeGain(1)(y_true, y_pred)
            0.0
            >>> round(DiscountedCumulativeGain(k=-1)(y_true, y_pred), 2)
            0.0
            >>> round(DiscountedCumulativeGain(k=2)(y_true, y_pred), 2)
            2.73
            >>> round(DiscountedCumulativeGain(k=3)(y_true, y_pred), 2)
            2.73
            >>> type(DiscountedCumulativeGain(k=1)(y_true, y_pred))
            <class 'float'>

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.

        :return: Discounted cumulative gain.
        """
        if self._k <= 0:
            return 0.
        coupled_pair = sort_and_couple(y_true, y_pred)
        result = 0.
        for i, (label, score) in enumerate(coupled_pair):
            if i >= self._k:
                break
            if label > self._threshold:
                result += (math.pow(2., label) - 1.) / math.log(2. + i)
        return result
