import pytest
import numpy as np
import shutil

from matchzoo import engine
from matchzoo import inputs


@pytest.fixture(scope='session')
def transformer():
    t = inputs.NaiveTransformer()
    t.guess_and_fill_missing_params()
    return t


@pytest.fixture(scope='session')
def x(transformer):
    return [("data is limited.", "life is hard.")]


@pytest.fixture(scope='session')
def y(transformer):
    return np.asarray([0])

def test_naive_transformer(transformer, x, y):
    assert transformer.fit(x)
    assert transformer.fit_transform(x, y) == [([0, 1, 2], [3, 1, 4])]
    assert transformer.fit(x).transform(x, y) == [([0, 1, 2], [3, 1, 4])]

def test_save_load_transformer(transformer):
    tmpfile = 'naivetransformer'
    transformer.save(tmpfile)
    assert engine.load_transformer(tmpfile)
    # shutil.rmtree(tmpfile)
