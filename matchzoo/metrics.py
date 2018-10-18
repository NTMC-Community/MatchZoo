"""Evaluation metrics for information retrieval."""

import math
import random
import numpy as np
import typing


def _to_list(x: typing.Any) -> list:
    """Transfer the inputs to list."""
    if isinstance(x, list):
        return x
    return [x]


def sort_couple(labels: list, scores: list) -> list:
    """Zip the `labels` with `scores` into a single list."""
    labels = _to_list(np.squeeze(labels).tolist())
    scores = _to_list(np.squeeze(scores).tolist())
    couple = list(zip(labels, scores))
    random.shuffle(couple)
    sorted_couple = sorted(couple, key=lambda x: x[1], reverse=True)
    return sorted_couple


def mean_reciprocal_rank(y_true: list,
                         y_pred: list,
                         rel_threshold: int=0) -> float:
    """
    Calculate reciprocal of the rank of the first relevant item.

    Example:
        >>> y_pred = [0.2, 0.3, 0.7, 1.0]
        >>> y_true = [1, 0, 0, 0]
        >>> mean_reciprocal_rank(y_true, y_pred)
        0.25

    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :param rel_threshold: the label threshold of relevance degree.
    :return: Mean reciprocal rank.
    """
    c = sort_couple(y_true, y_pred)
    for j, (label, pred) in enumerate(c):
        if label > rel_threshold:
            return 1. / (j + 1)
    return 0.


def precision_at_k(y_true: list,
                   y_pred: list,
                   k: int,
                   rel_threshold: int=0) -> float:
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
    :param rel_threshold: the label threshold of relevance degree.
    :return: Precision @ k
    :raises: ValueError: len(r) must be >= k.
    """
    if k <= 0:
        raise ValueError('k must be larger than 0.')
    c = sort_couple(y_true, y_pred)
    precision = 0.0
    for i, (label, score) in enumerate(c):
        if i >= k:
            break
        if label > rel_threshold:
            precision += 1.
    return precision / k


def average_precision(y_true: list,
                      y_pred: list,
                      rel_threshold: int=0) -> float:
    """
    Calculate average precision (area under PR curve).

    Example:
        >>> y_true = [0, 1, 0, 0]
        >>> y_pred = [0.1, 0.6, 0.2, 0.3]
        >>> round(average_precision(y_true, y_pred), 2)
        0.52

    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :param rel_threshold: the label threshold of relevance degree.
    :return: Average precision.
    """
    out = [precision_at_k(y_true, y_pred, k + 1) for k in range(len(y_pred))]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(y_true: list,
                           y_pred: list,
                           rel_threshold: int=0) -> float:
    """
    Calculate mean average precision.

    Example:
        >>> y_true = [0, 1, 0, 0]
        >>> y_pred = [0.1, 0.6, 0.2, 0.3]
        >>> mean_average_precision(y_true, y_pred)
        1.0

    :param y_true: The ground true label of each document.
    :param y_pred: The predicted scores of each document.
    :param rel_threshold: the label threshold of relevance degree.
    :return: Mean average precision.
    """
    s = 0.
    ipos = 0
    c = sort_couple(y_true, y_pred)
    for j, (label, score) in enumerate(c):
        if label > rel_threshold:
            ipos += 1.
            s += ipos / (j + 1.)
    if ipos == 0:
        return 0.
    else:
        return s/ipos


def dcg_at_k(y_true: list,
             y_pred: list,
             k: int,
             rel_threshold: int=0) -> float:
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
    :param rel_threshold: the label threshold of relevance degree.

    :return: Discounted cumulative gain.
    """
    if k <= 0.:
        return 0.
    c = sort_couple(y_true, y_pred)
    s = 0.
    for i, (label, score) in enumerate(c):
        if i >= k:
            break
        if label > rel_threshold:
            s += (math.pow(2., label) - 1.) / math.log(2. + i)
    return s


def ndcg_at_k(y_true: list,
              y_pred: list,
              k: int,
              rel_threshold: int=0) -> float:
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
    :param rel_threshold: the label threshold of relevance degree.

    :return: Normalized discounted cumulative gain.
    """
    idcg = dcg_at_k(y_true, y_true, k, rel_threshold)
    dcg = dcg_at_k(y_true, y_pred, k, rel_threshold)
    return dcg / idcg
