"""
These tests are simplied because the original verion takes too much time to
run, making CI fails as it reaches the time limit.
"""

import pytest
import copy
from pathlib import Path
import shutil

import matchzoo as mz
from keras.backend import clear_session

@pytest.fixture(scope='module', params=[
    mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=2)),
    mz.tasks.Classification(num_classes=2),
])
def task(request):
    return request.param


@pytest.fixture(scope='module')
def train_raw(task):
    return mz.datasets.toy.load_data('train', task)[:5]


@pytest.fixture(scope='module', params=mz.models.list_available())
def model_class(request):
    return request.param


@pytest.fixture(scope='module')
def embedding():
    return mz.datasets.toy.load_embedding()


@pytest.fixture(scope='module')
def setup(task, model_class, train_raw, embedding):
    clear_session() # prevent OOM during CI tests
    return mz.auto.prepare(
        task=task,
        model_class=model_class,
        data_pack=train_raw,
        embedding=embedding
    )


@pytest.fixture(scope='module')
def model(setup):
    return setup[0]


@pytest.fixture(scope='module')
def preprocessor(setup):
    return setup[1]


@pytest.fixture(scope='module')
def gen_builder(setup):
    return setup[2]


@pytest.fixture(scope='module')
def embedding_matrix(setup):
    return setup[3]


@pytest.fixture(scope='module')
def data(train_raw, preprocessor, gen_builder):
    return gen_builder.build(preprocessor.transform(train_raw))[0]


@pytest.mark.slow
def test_model_fit_eval_predict(model, data):
    x, y = data
    batch_size = len(x['id_left'])
    assert model.fit(x, y, batch_size=batch_size, verbose=0)
    assert model.evaluate(x, y, batch_size=batch_size)
    assert model.predict(x, batch_size=batch_size) is not None


@pytest.mark.cron
def test_save_load_model(model):
    tmpdir = '.matchzoo_test_save_load_tmpdir'

    if Path(tmpdir).exists():
        shutil.rmtree(tmpdir)

    try:
        model.save(tmpdir)
        assert mz.load_model(tmpdir)
        with pytest.raises(FileExistsError):
            model.save(tmpdir)
    finally:
        if Path(tmpdir).exists():
            shutil.rmtree(tmpdir)


@pytest.mark.cron
def test_hyper_space(model):
    for _ in range(2):
        new_params = copy.deepcopy(model.params)
        sample = mz.hyper_spaces.sample(new_params.hyper_space)
        for key, value in sample.items():
            new_params[key] = value
        new_model = new_params['model_class'](params=new_params)
        new_model.build()
        new_model.compile()
