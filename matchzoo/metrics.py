"""Evaluation metrics for information retrieval."""
import math
import random
import numpy as np

from matchzoo import engine


def sort_and_couple(labels: list, scores: np.array) -> list:
    """Zip the `labels` with `scores` into a single list."""
    labels = np.squeeze(labels).tolist()
    scores = np.squeeze(scores).tolist()
    couple = list(zip(labels, scores))
    random.shuffle(couple)
    sorted_couple = sorted(couple, key=lambda x: x[1], reverse=True)
    return sorted_couple


class MeanReciprocalRank(engine.BaseMetric):
    ALIAS = ['mean_reciprocal_rank', 'mrr', 'MRR']

    def __init__(self, threshold=0):
        """
        :param threshold: the label threshold of relevance degree.
        """
        self._threshold = threshold

    def __repr__(self):
        return f'{self.ALIAS[0]}({self._threshold})'

    def __call__(self, y_true, y_pred):
        """
        Calculate reciprocal of the rank of the first relevant item.

        Example:
            >>> y_pred = np.asarray([0.2, 0.3, 0.7, 1.0])
            >>> y_true = np.asarray([1, 0, 0, 0])
            >>> MeanReciprocalRank()(y_true, y_pred)
            0.25

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Mean reciprocal rank.
        """
        coupled_pair = sort_and_couple(y_true, y_pred)
        for idx, (label, pred) in enumerate(coupled_pair):
            if label > self._threshold:
                return 1. / (idx + 1)
        return 0.


class Precision(engine.BaseMetric):
    ALIAS = 'precision'

    def __init__(self, k=1, threshold=0):
        self._k = k
        self._threshold = threshold

    def __repr__(self):
        return f"{self.ALIAS}@{self._k}({self._threshold})"

    def __call__(self, y_true, y_pred):
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
            raise ValueError('self._k must be larger than 0.')
        coupled_pair = sort_and_couple(y_true, y_pred)
        precision = 0.0
        for idx, (label, score) in enumerate(coupled_pair):
            if idx >= self._k:
                break
            if label > self._threshold:
                precision += 1.
        return precision / self._k


class AveragePrecision(engine.BaseMetric):
    ALIAS = 'average_precision'

    def __init__(self, threshold=0):
        self._threshold = threshold

    def __repr__(self):
        return f"{self.ALIAS}({self._threshold})"

    def __call__(self, y_true, y_pred):
        """
        Calculate average precision (area under PR curve).

        Example:
            >>> y_true = [0, 1]
            >>> y_pred = [0.1, 0.6]
            >>> round(AveragePrecision()(y_true, y_pred), 2)
            0.75

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Average precision.
        """
        precision_metrics = [Precision(k + 1) for k in range(len(y_pred))]
        out = [metric(y_true, y_pred) for metric in precision_metrics]
        if not out:
            return 0.
        return np.mean(out)


class MeanAveragePrecision(engine.BaseMetric):
    ALIAS = ['mean_average_precision', 'map', 'MAP']

    def __init__(self, threshold=0):
        self._threshold = threshold

    def __repr__(self):
        return f"{self.ALIAS[0]}({self._threshold})"

    def __call__(self, y_true, y_pred):
        """
        Calculate mean average precision.

        Example:
            >>> y_true = [0, 1, 0, 0]
            >>> y_pred = [0.1, 0.6, 0.2, 0.3]
            >>> MeanAveragePrecision()(y_true, y_pred)
            1.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Mean average precision.
        """
        result = 0.
        pos = 0
        coupled_pair = sort_and_couple(y_true, y_pred)
        for idx, (label, score) in enumerate(coupled_pair):
            if label > self._threshold:
                pos += 1.
                result += pos / (idx + 1.)
        if pos == 0:
            return 0.
        else:
            return result / pos


class DiscountedCumulativeGain(engine.BaseMetric):
    ALIAS = ['discounted_cumulative_gain', 'dcg', 'DCG']

    def __init__(self, k=1, threshold=0):
        """
        :param k: Number of results to consider
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self):
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def __call__(self, y_true, y_pred):
        """
        Calculate discounted cumulative gain (dcg).

        Relevance is positive real values or binary values.

        Example:
            >>> y_true = [0, 1, 2, 0]
            >>> y_pred = [0.4, 0.2, 0.5, 0.7]
            >>> DiscountedCumulativeGain(1)(y_true, y_pred)
            0.0
            >>> round(DiscountedCumulativeGain(2)(y_true, y_pred), 2)
            2.73
            >>> round(DiscountedCumulativeGain(3)(y_true, y_pred), 2)
            2.73
            >>> type(DiscountedCumulativeGain(1)(y_true, y_pred))
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


class NormalizedDiscountedCumulativeGain(engine.BaseMetric):
    ALIAS = ['normalized_discounted_cumulative_gain', 'ndcg', 'NDCG']

    def __init__(self, k=1, threshold=0):
        """
        :param k: Number of results to consider
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self):
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def __call__(self, y_true, y_pred):
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
        dcg_metric = DiscountedCumulativeGain(
                k=self._k, threshold=self._threshold)
        idcg_val = dcg_metric(y_true, y_true)
        dcg_val = dcg_metric(y_true, y_pred)
        return dcg_val / idcg_val
