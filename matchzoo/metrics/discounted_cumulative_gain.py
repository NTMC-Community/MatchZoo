"""Discounted cumulative gain metric for ranking."""
import math

import numpy as np

from matchzoo import engine


class DiscountedCumulativeGain(engine.BaseMetric):
    """Disconunted cumulative gain metric."""

    ALIAS = ['discounted_cumulative_gain', 'dcg']

    def __init__(self, k: int = 1):
        """
        :class:`DiscountedCumulativeGain` constructor.

        :param k: Number of results to consider.
        """
        self._k = k

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self._k}"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate discounted cumulative gain (dcg).

        Relevance is positive real values or binary values.

        Example:
            >>> y_true = []
            >>> y_pred = [0.4, 0.2, 0.5, 0.7]
            >>> DiscountedCumulativeGain(0)(y_true, y_pred)
            0.0
            >>> round(DiscountedCumulativeGain(k=1)(y_true, y_pred), 2)
            0.4
            >>> round(DiscountedCumulativeGain(k=2)(y_true, y_pred), 2)
            0.6
            >>> round(DiscountedCumulativeGain(k=3)(y_true, y_pred), 2)
            0.92

        :param y_true: The relevance degree label of each document.
        :param y_pred: The predicted scores of each document.

        :return: Discounted cumulative gain.
        """
        y_pred = np.asfarray(y_pred)[:self._k]
        if self._k <= 0:
            return 0.
        elif self._k == 1:
            return y_pred[0]
        else:
            return y_pred[0] + np.sum(y_pred[1:] / np.log2(
                np.arange(2, y_pred.size + 1)))
