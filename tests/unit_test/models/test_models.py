import pytest
import copy
from pathlib import Path
import shutil

import matchzoo as mz


@pytest.fixture(scope='module', params=[
    mz.tasks.Ranking(),
    mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=2)),
    mz.tasks.Classification(num_classes=2)
])
def task(request):
    return request.param


@pytest.fixture(scope='module')
def train_raw(task):
    return mz.datasets.toy.load_data('train', task)


@pytest.fixture(scope='module')
def dev_raw(task):
    return mz.datasets.toy.load_data('dev', task)


@pytest.fixture(scope='module')
def test_raw(task):
    return mz.datasets.toy.load_data('test', task)


@pytest.fixture(scope='module', params=mz.models.list_available())
def model_class(request):
    return request.param


@pytest.fixture(scope='module', params=[
    None,
    mz.datasets.embeddings.load_glove_embedding(dimension=50),
])
def embedding(request):
    return request.param


@pytest.fixture(scope='module')
def setup(task, model_class, train_raw, embedding):
    keeper = mz.ZooKeeper(task)
    return keeper.prepare(
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
def train_gen(train_raw, preprocessor, gen_builder):
    return gen_builder.build(preprocessor.transform(train_raw, verbose=0))


@pytest.fixture(scope='module')
def dev_gen(train_raw, preprocessor, gen_builder):
    return gen_builder.build(preprocessor.transform(dev_raw, verbose=0))


@pytest.fixture(scope='module')
def test_gen(test_raw, preprocessor, gen_builder):
    return gen_builder.build(preprocessor.transform(test_raw, verbose=0))


@pytest.mark.slow
def test_model_fit(model, train_gen):
    x, y = train_gen[0]
    assert model.fit(x, y, verbose=0)


@pytest.mark.slow
def test_model_fit_generator(model, train_gen):
    assert model.fit_generator(train_gen, verbose=0)


@pytest.mark.slow
def test_model_evaluate(model, dev_gen):
    x, y = dev_gen[0]
    assert model.evaluate(x, y)


@pytest.mark.slow
def test_model_evaluate_generator(model, dev_gen):
    assert model.evaluate(dev_gen)


@pytest.mark.slow
def test_model_predict(model, test_gen):
    x, y = test_gen[0]
    assert model.predict(x) is not None


@pytest.mark.slow
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


@pytest.mark.slow
def test_tuner(model, train_gen, test_gen):
    tuner = mz.auto.Tuner(
        params=model.params,
        train_data=train_gen,
        test_data=test_gen,
        fit_kwargs=dict(epochs=1),
        num_runs=1
    )
    if 'with_embedding' in model.params:
        tuner.callbacks.append(
            mz.tuner.callbacks.LoadEmbeddingMatrix(embedding_matrix))
    assert tuner.tune()


@pytest.mark.slow
def test_hyper_space(model):
    for _ in range(16):
        new_params = copy.deepcopy(model.params)
        sample = mz.hyper_spaces.sample(new_params.hyper_space)
        for key, value in sample.items():
            new_params[key] = value
        new_model = new_params['model_class'](params=new_params)
        new_model.build()
        new_model.compile()
