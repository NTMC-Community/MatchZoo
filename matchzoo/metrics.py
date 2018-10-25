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
    ALIAS = 'mean_reciprocal_rank'

    def __init__(self, threshold=0):
        """
        :param threshold: the label threshold of relevance degree.
        """
        self._threshold = threshold

    def __repr__(self):
        return self.ALIAS + '(' + str(self._threshold) + ')'

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


def precision_at_k(y_true: list,
                   y_pred: list,
                   k: int,
                   threshold: int = 0) -> float:
    """
    Calculate precision@k.

    Example:
        >>> y_true = [0, 0, 0, 1]
        >>> y_pred = [0.2, 0.4, 0.3, 0.1]
        >>> precision_at_k(y_true, y_pred, 1)
        0.0
        >>> precision_at_k(y_true, y_pred, 2)
        0.0
        >>> precision_at_k(y_true, y_pred, 4)
        0.25
        >>> precision_at_k(y_true, y_pred, 5)
        0.2

    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :param threshold: the label threshold of relevance degree.
    :return: Precision @ k
    :raises: ValueError: len(r) must be >= k.
    """
    if k <= 0:
        raise ValueError('k must be larger than 0.')
    coupled_pair = sort_and_couple(y_true, y_pred)
    precision = 0.0
    for idx, (label, score) in enumerate(coupled_pair):
        if idx >= k:
            break
        if label > threshold:
            precision += 1.
    return precision / k


def average_precision(y_true: list,
                      y_pred: list,
                      threshold: int = 0) -> float:
    """
    Calculate average precision (area under PR curve).

    Example:
        >>> y_true = [0, 1]
        >>> y_pred = [0.1, 0.6]
        >>> round(average_precision(y_true, y_pred), 2)
        0.75

    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :param threshold: the label threshold of relevance degree.
    :return: Average precision.
    """
    out = [precision_at_k(y_true, y_pred, k + 1) for k in range(len(y_pred))]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(y_true: list,
                           y_pred: list,
                           threshold: int = 0) -> float:
    """
    Calculate mean average precision.

    Example:
        >>> y_true = [0, 1, 0, 0]
        >>> y_pred = [0.1, 0.6, 0.2, 0.3]
        >>> mean_average_precision(y_true, y_pred)
        1.0

    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :param threshold: the label threshold of relevance degree.
    :return: Mean average precision.
    """
    result = 0.
    pos = 0
    coupled_pair = sort_and_couple(y_true, y_pred)
    for idx, (label, score) in enumerate(coupled_pair):
        if label > threshold:
            pos += 1.
            result += pos / (idx + 1.)
    if pos == 0:
        return 0.
    else:
        return result / pos


def dcg_at_k(y_true: list,
             y_pred: list,
             k: int,
             threshold: int = 0) -> float:
    """
    Calculate discounted cumulative gain (dcg).

    Relevance is positive real values or binary values.

    Example:
        >>> y_true = [0, 1, 2, 0]
        >>> y_pred = [0.4, 0.2, 0.5, 0.7]
        >>> dcg_at_k(y_true, y_pred, 1)
        0.0
        >>> round(dcg_at_k(y_true, y_pred, 2), 2)
        2.73
        >>> round(dcg_at_k(y_true, y_pred, 3), 2)
        2.73
        >>> type(dcg_at_k(y_true, y_pred, 1))
        <class 'float'>

    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :param k: Number of results to consider
    :param threshold: the label threshold of relevance degree.

    :return: Discounted cumulative gain.
    """
    if k <= 0.:
        return 0.
    coupled_pair = sort_and_couple(y_true, y_pred)
    result = 0.
    for i, (label, score) in enumerate(coupled_pair):
        if i >= k:
            break
        if label > threshold:
            result += (math.pow(2., label) - 1.) / math.log(2. + i)
    return result


def ndcg_at_k(y_true: list,
              y_pred: list,
              k: int,
              threshold: int = 0) -> float:
    """
    Calculate normalized discounted cumulative gain (ndcg).

    Relevance is positive real values or binary values.

    Example:
        >>> y_true = [0, 1, 2, 0]
        >>> y_pred = [0.4, 0.2, 0.5, 0.7]
        >>> ndcg_at_k(y_true, y_pred, 1)
        0.0
        >>> round(ndcg_at_k(y_true, y_pred, 2), 2)
        0.52
        >>> round(ndcg_at_k(y_true, y_pred, 3), 2)
        0.52
        >>> type(ndcg_at_k(y_true, y_pred, 1))
        <class 'float'>

    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :param k: Number of results to consider
    :param threshold: the label threshold of relevance degree.

    :return: Normalized discounted cumulative gain.
    """
    idcg = dcg_at_k(y_true, y_true, k, threshold)
    dcg = dcg_at_k(y_true, y_pred, k, threshold)
    return dcg / idcg
