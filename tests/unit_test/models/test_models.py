import numpy as np
from pathlib import Path
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
# with no kwargs: (models.DenseBaseline, None)
# with kwargs: (models.DenseBaseline, {"num_dense_units": 512})
model_setups = [
    (models.Naive, None, [np.float32, np.float32]),
    (models.DenseBaseline, None, [np.float32, np.float32]),
    (models.DSSM, None, [np.float32, np.float32]),
    (models.CDSSM, None, [np.float32, np.float32]),
    (models.ArcI, None, [np.int32, np.int32]),
    (models.ArcII, None, [np.int32, np.int32])
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
    model_class, custom_kwargs, input_dtypes = request.param
    model = model_class()
    if custom_kwargs:
        for key, val in custom_kwargs.items():
            model.params[key] = val
    return model, input_dtypes


@pytest.fixture
def compiled_model(raw_model, task):
    model, input_dtypes = raw_model
    model.params['task'] = task
    model.guess_and_fill_missing_params()
    model.build()
    model.compile()
    return model, input_dtypes


@pytest.fixture
def x(compiled_model, num_samples):
    model, input_dtypes = compiled_model
    rand_func = {np.float32:
                     lambda x: np.random.uniform(low=-1, high=1, size=x),
                 np.int32:
                     lambda x: np.random.randint(low=0, high=100, size=x)
                 }
    input_shapes = model.params['input_shapes']

    values = [rand_func[dtype]([num_samples] + list(shape))
            if None not in shape
            else rand_func[dtype]([num_samples] + [10, 900])
            for shape, dtype
            in zip(input_shapes, input_dtypes)]
    return {'text_left': values[0],
            'text_right': values[1],
            'id_left': np.random.randint(low=0, high=1000, size=[num_samples])
            }

@pytest.fixture
def y(compiled_model, num_samples):
    model, input_dtypes = compiled_model
    task = model.params['task']
    return np.random.randn(num_samples, *task.output_shape)

@pytest.mark.slow
def test_get_default_preprocessor(raw_model):
    model, _ = raw_model
    assert model.get_default_preprocessor()

@pytest.mark.slow
def test_model_fit(compiled_model, x, y):
    model, input_dtypes = compiled_model
    assert model.fit(x, y, verbose=0)


@pytest.mark.slow
def test_model_evaluate(compiled_model, x, y):
    model, input_dtypes = compiled_model
    assert model.evaluate(x, y, verbose=0)


@pytest.mark.slow
def test_model_predict(compiled_model, x):
    model, input_dtypes = compiled_model
    assert model.predict(x) is not None


@pytest.mark.slow
def test_save_load_model(compiled_model):
    model, input_dtypes = compiled_model
    tmpdir = '.tmpdir'
    if Path(tmpdir).exists():
        shutil.rmtree(tmpdir)
    model.save(tmpdir)
    assert engine.load_model(tmpdir)
    with pytest.raises(FileExistsError):
        model.save(tmpdir)
    shutil.rmtree(tmpdir)
