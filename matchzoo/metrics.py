"""Evaluation metrics for information retrieval."""

import numpy as np


def mean_reciprocal_rank(rs: list) -> float:
    """
    Calculate reciprocal of the rank of the first relevant item.

    First element is 'rank 1'. Relevance is binary (nonzero is relevant).

    Example:
        >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        >>> round(mean_reciprocal_rank(rs), 2)
        0.61
        >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        >>> mean_reciprocal_rank(rs)
        0.5
        >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
        >>> mean_reciprocal_rank(rs)
        0.75

    :param rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item).
    :return: Mean reciprocal rank.
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r: list) -> float:
    """
    Calculate precision after all relevant documents have been retrieved.

    Relevance is binary (nonzero is relevant).

    Example:
        >>> r = [0, 0, 1]
        >>> round(r_precision(r), 2)
        0.33
        >>> r = [0, 1, 0]
        >>> r_precision(r)
        0.5
        >>> r = [1, 0, 0]
        >>> r_precision(r)
        1.0
        >>> r = [0, 0, 0]
        >>> r_precision(r)
        0.0

    :param r: Relevance scores (list or numpy) in rank order
              (first element is the first item).
    :return: Precision score.
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r: list, k: int) -> float:
    """
    Calculate precision@k.

    Relevance is binary (nonzero is relevant).

    Example:
        >>> r = [0, 0, 1]
        >>> precision_at_k(r, 1)
        0.0
        >>> precision_at_k(r, 2)
        0.0
        >>> round(precision_at_k(r, 3), 2)
        0.33
        >>> precision_at_k(r, 4)
        Traceback (most recent call last):
            File "<stdin>", line 1, in ?
        ValueError: Relevance score length < k


    :param r: Relevance scores (list or numpy) in rank order
              (first element is the first item)
    :return: Precision @ k
    :raises: ValueError: len(r) must be >= k.
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r: list):
    """
    Calculate average precision (area under PR curve).

    Relevance is binary (nonzero is relevant).

    Example:
        >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
        >>> round(average_precision(r), 2)
        0.78

    :param r: Relevance scores (list or numpy) in rank order
              (first element is the first item).
    :return: Average precision.
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs: list) -> float:
    """
    Calculate mean average precision.

    Relevance is binary (nonzero is relevant).

    Example:
        >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
        >>> round(mean_average_precision(rs), 2)
        0.78
        >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
        >>> round(mean_average_precision(rs), 2)
        0.39

    :param rs: Iterator of relevance scores (list or numpy) in rank order
               (first element is the first item).
    :return: Mean average precision.
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r: list, k: int, method: int=0) -> float:
    """
    Calculate discounted cumulative gain (dcg).

    Relevance is positive real values or binary values.

    Example:
        >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
        >>> dcg_at_k(r, 1)
        3.0
        >>> dcg_at_k(r, 1, method=1)
        3.0
        >>> dcg_at_k(r, 2)
        5.0
        >>> round(dcg_at_k(r, 2, method=1), 2)
        4.26
        >>> round(dcg_at_k(r, 10), 2)
        9.61
        >>> round(dcg_at_k(r, 11), 2)
        9.61

    :param r: Relevance scores (list or numpy) in rank order
              (first element is the first item).
    :param k: Number of results to consider

    :param method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                   If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...].

    :return: Discounted cumulative gain.
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        else:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r: list, k: int, method: int=0) -> float:
    """
    Calculate normalized discounted cumulative gain (ndcg).

    Relevance is positive real values or binary values.

    Example:
        >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
        >>> ndcg_at_k(r, 1)
        1.0
        >>> r = [2, 1, 2, 0]
        >>> round(ndcg_at_k(r, 4), 2)
        0.92
        >>> round(ndcg_at_k(r, 4, method=1), 2)
        0.97
        >>> ndcg_at_k([0], 1)
        0.0
        >>> ndcg_at_k([1], 2)
        1.0

    :param r: Relevance scores (list or numpy) in rank order
              (first element is the first item).
    :param k: Number of results to consider

    :param method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                   If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...].

    :return: Normalized discounted cumulative gain.
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max
