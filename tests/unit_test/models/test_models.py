import numpy as np
import shutil

import pytest

from matchzoo import engine
from matchzoo import models
from matchzoo import tasks

# To add a test for a new model, add a tuple of form:
#       (model_class, customized_kwargs)
# If no customized_kwargs is needed, simply put a `None`.
# Notice that each of such tuple will go through a full testing procedure, so
# it's quite time consuming. Don't add customized_kwargs unless you have to.
# Examples:
# with no kwargs: (models.DenseBaselineModel, None)
# with kwargs: (models.DenseBaselineModel, {"num_dense_units": 512})
model_setups = [
    (models.NaiveModel, None),
    (models.DenseBaselineModel, None),
    (models.DSSMModel, None)
]


@pytest.fixture(scope='module', params=[1, 32])
def num_samples(request):
    return request.param


@pytest.fixture(scope='module', params=[
    tasks.Classification(num_classes=2),
    tasks.Classification(num_classes=16),
    tasks.Ranking()
])
def task(request):
    return request.param


@pytest.fixture(params=model_setups)
def raw_model(request):
    model_class, custom_kwargs = request.param
    model = model_class()
    if custom_kwargs:
        for key, val in custom_kwargs.items():
            model.params[key] = val
    return model


@pytest.fixture
def compiled_model(raw_model, task):
    raw_model.params['task'] = task
    raw_model.guess_and_fill_missing_params()
    raw_model.build()
    raw_model.compile()
    return raw_model


@pytest.fixture
def x(compiled_model, num_samples):
    input_shapes = compiled_model.params['input_shapes']
    return [np.random.randn(num_samples, *shape) for shape in input_shapes]


@pytest.fixture
def y(compiled_model, num_samples):
    task = compiled_model.params['task']
    return np.random.randn(num_samples, *task.output_shape)


def test_model_fit(compiled_model, x, y):
    assert compiled_model.fit(x, y, verbose=0)


def test_model_evaluate(compiled_model, x, y):
    assert compiled_model.evaluate(x, y, verbose=0)


def test_model_predict(compiled_model, x):
    assert compiled_model.predict(x) is not None


def test_save_load_model(compiled_model):
    tmpdir = '.tmpdir'
    compiled_model.save(tmpdir)
    assert engine.load_model(tmpdir)
    with pytest.raises(FileExistsError):
        compiled_model.save(tmpdir)
    shutil.rmtree(tmpdir)
