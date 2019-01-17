"""Mean reciprocal ranking metric."""
import numpy as np

from matchzoo import engine


class MeanReciprocalRank(engine.BaseMetric):
    """Mean reciprocal rank metric."""

    ALIAS = ['mean_reciprocal_rank', 'mrr']

    def __init__(self, threshold: float = 0.):
        """
        :class:`MeanReciprocalRankMetric`.

        :param threshold: The label threshold of relevance degree.
        """
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f'{self.ALIAS[0]}({self._threshold})'

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate reciprocal of the rank of the first relevant item.

        Example:
            >>> import numpy as np
            >>> y_pred = np.asarray([0.2, 0.3, 0.7, 1.0])
            >>> y_true = np.asarray([1, 0, 0, 0])
            >>> MeanReciprocalRank()(y_true, y_pred)
            0.25

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Mean reciprocal rank.
        """
        coupled_pair = engine.sort_and_couple(y_true, y_pred)
        for idx, (label, pred) in enumerate(coupled_pair):
            if label > self._threshold:
                return 1. / (idx + 1)
        return 0.
