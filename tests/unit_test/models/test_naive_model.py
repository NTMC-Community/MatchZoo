import numpy as np
import shutil

import pytest

from matchzoo import engine
from matchzoo import models
from matchzoo import tasks


@pytest.fixture(scope='module')
def model():
    m = models.NaiveModel()
    m.params['task'] = tasks.Classification(num_classes=2)
    m.guess_and_fill_missing_params()
    m.build()
    m.compile()
    return m


@pytest.fixture(scope='module')
def x(model):
    return [np.random.randn(1, *model.params['input_shapes'][0]),
            np.random.randn(1, *model.params['input_shapes'][1])]


@pytest.fixture(scope='module')
def y(model):
    return np.asarray([[0, 1]])


def test_naive_model(model, x, y):
    assert model.fit(x, y)
    assert model.evaluate(x, y)
    assert model.predict(x) is not None


def test_save_load_model(model):
    tmpdir = '.tmpdir'
    model.save(tmpdir)
    assert engine.load_model(tmpdir)
    with pytest.raises(FileExistsError):
        model.save(tmpdir)
    shutil.rmtree(tmpdir)
