import pytest

from matchzoo import metrics


def test_to_list():
    x = [1, 2, 3]
    assert metrics._to_list(x) == [1, 2, 3]

def test_sort_couple():
    l = [0, 1, 2]
    s = [0.1, 0.4, 0.2]
    c = metrics.sort_couple(l, s)
    assert c == [(1, 0.4), (2, 0.2), (0, 0.1)]

def test_mean_reciprocal_rank():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert metrics.mean_reciprocal_rank(label, score) == 1

def test_precision_at_k():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert metrics.precision_at_k(label, score, 1) == 1.
    assert metrics.precision_at_k(label, score, 2) == 1.
    assert round(metrics.precision_at_k(label, score, 3), 2) == 0.67

def test_average_precision():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert round(metrics.average_precision(label, score), 2) == 0.89

def test_mean_average_precision():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert metrics.mean_average_precision(label, score) == 1.

def test_dcg_at_k():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert round(metrics.dcg_at_k(label, score, 1), 2) == 1.44
    assert round(metrics.dcg_at_k(label, score, 2), 2) == 4.17
    assert round(metrics.dcg_at_k(label, score, 3), 2) == 4.17

def test_ndcg_at_k():
    label = [0, 1, 2]
    score = [0.1, 0.4, 0.2]
    assert round(metrics.ndcg_at_k(label, score, 1), 2) == 0.33
    assert round(metrics.ndcg_at_k(label, score, 2), 2) == 0.80
    assert round(metrics.ndcg_at_k(label, score, 3), 2) == 0.80
