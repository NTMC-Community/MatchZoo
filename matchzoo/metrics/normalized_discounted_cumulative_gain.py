"""Normalized discounted cumulative gain metric for ranking."""
import numpy as np

from matchzoo import engine
from .discounted_cumulative_gain import DiscountedCumulativeGain


class NormalizedDiscountedCumulativeGain(engine.BaseMetric):
    """Normalized discounted cumulative gain metric."""

    ALIAS = ['normalized_discounted_cumulative_gain', 'ndcg']

    def __init__(self, k: int = 1, threshold: float = 0.):
        """
        :class:`NormalizedDiscountedCumulativeGain` constructor.

        :param k: Number of results to consider
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate normalized discounted cumulative gain (ndcg).

        Relevance is positive real values or binary values.

        Example:
            >>> y_true = [0, 1, 2, 0]
            >>> y_pred = [0.4, 0.2, 0.5, 0.7]
            >>> ndcg = NormalizedDiscountedCumulativeGain
            >>> ndcg(k=1)(y_true, y_pred)
            0.0
            >>> round(ndcg(k=2)(y_true, y_pred), 2)
            0.52
            >>> round(ndcg(k=3)(y_true, y_pred), 2)
            0.52
            >>> type(ndcg()(y_true, y_pred))
            <class 'float'>

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.

        :return: Normalized discounted cumulative gain.
        """
        dcg_metric = DiscountedCumulativeGain(k=self._k,
                                              threshold=self._threshold)
        idcg_val = dcg_metric(y_true, y_true)
        dcg_val = dcg_metric(y_true, y_pred)
        return dcg_val / idcg_val if idcg_val != 0 else 0
