import os
import pytest
import shutil
import numpy as np

from matchzoo import datapack
from matchzoo import generators
from matchzoo import preprocessor
from matchzoo import models
from matchzoo import engine
from matchzoo import tasks


@pytest.fixture
def train():
    train = []
    path = os.path.dirname(__file__)
    with open(os.path.join(path, '../sample/train_rank.txt')) as f:
        train = [tuple(map(str, i.strip().split('\t'))) for i in f]
    return train


@pytest.fixture
def test():
    test = []
    path = os.path.dirname(__file__)
    with open(os.path.join(path, '../sample/test_rank.txt')) as f:
        test = [tuple(map(str, i.strip().split('\t'))) for i in f]
    return test


@pytest.fixture
def task(request) -> engine.BaseTask:
    return tasks.Ranking()


@pytest.fixture
def cdssm_preprocessor():
    return preprocessor.CDSSMPreprocessor()


@pytest.fixture
def processed_train(train, cdssm_preprocessor) -> datapack.DataPack:
    preprocessed_train = cdssm_preprocessor.fit_transform(train, stage='train')
    cdssm_preprocessor.save('.tmpdir')
    return preprocessed_train


@pytest.fixture
def processed_test(test) -> datapack.DataPack:
    cdssm_preprocessor = engine.load_preprocessor('.tmpdir')
    return cdssm_preprocessor.fit_transform(test, stage='test')


@pytest.fixture(params=['point', 'pair'])
def train_generator(request, processed_train, task) -> engine.BaseGenerator:
    if request.param == 'point':
        return generators.PointGenerator(processed_train,
                                         task=task,
                                         stage='train')
    elif request.param == 'pair':
        return generators.PairGenerator(processed_train, stage='train')


@pytest.fixture(params=['point', 'list'])
def test_generator(request, processed_test, task) -> engine.BaseGenerator:
    if request.param == 'point':
        return generators.PointGenerator(processed_test, task=task, stage='test')
    elif request.param == 'list':
        return generators.ListGenerator(processed_test, stage='test')


def test_cdssm(processed_train, task, train_generator, test_generator):
    """Test CDSSM model."""
    # Create a dssm model
    cdssm_model = models.CDSSMModel()
    cdssm_model.params['input_shapes'] = processed_train.context['input_shapes']
    cdssm_model.params['task'] = task
    cdssm_model.guess_and_fill_missing_params()
    cdssm_model.build()
    cdssm_model.compile()
    cdssm_model.fit_generator(train_generator)
    # save
    cdssm_model.save('.tmpdir')

    # testing
    X, y = test_generator[0]
    dssm_model = engine.load_model('.tmpdir')
    predictions = dssm_model.predict([X.text_left, X.text_right])
    assert len(predictions) > 0
    assert type(predictions[0][0]) == np.float32
    shutil.rmtree('.tmpdir')