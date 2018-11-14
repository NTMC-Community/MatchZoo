import os
import shutil

import numpy as np
import pytest

from data_pack import data_pack
from matchzoo import engine
from matchzoo import generators
from matchzoo import models
from matchzoo import preprocessors
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
def dssm_preprocessor():
    return preprocessors.DSSMPreprocessor()


@pytest.fixture
def processed_train(train, dssm_preprocessor) -> data_pack.DataPack:
    preprocessed_train = dssm_preprocessor.fit_transform(train, stage='train')
    dssm_preprocessor.save('.tmpdir')
    return preprocessed_train


@pytest.fixture
def processed_test(test) -> data_pack.DataPack:
    dssm_proprecessor = engine.load_preprocessor('.tmpdir')
    return dssm_proprecessor.fit_transform(test, stage='predict')


@pytest.fixture(params=['point', 'pair'])
def train_generator(request, processed_train, task) -> engine.DataGenerator:
    if request.param == 'point':
        return generators.PointGenerator(processed_train,
                                         task=task,
                                         stage='train')
    elif request.param == 'pair':
        return generators.PairGenerator(processed_train,
                                        stage='train')


@pytest.fixture(params=['point', 'list'])
def test_generator(request, processed_test, task) -> engine.DataGenerator:
    if request.param == 'point':
        return generators.PointGenerator(processed_test, task=task,
                                         stage='predict')
    elif request.param == 'list':
        return generators.ListGenerator(processed_test, stage='predict')


@pytest.mark.slow
def test_dssm(processed_train,
              task,
              train_generator,
              test_generator):
    """Test DSSM model."""
    # Create a dssm model
    dssm_model = models.DSSMModel()
    dssm_model.params['input_shapes'] = processed_train.context['input_shapes']
    dssm_model.params['task'] = task
    dssm_model.guess_and_fill_missing_params()
    dssm_model.build()
    dssm_model.compile()
    dssm_model.fit_generator(train_generator)
    # save
    dssm_model.save('.tmpdir')

    # testing
    X, y = test_generator[0]
    dssm_model = engine.load_model('.tmpdir')
    predictions = dssm_model.predict([X.text_left, X.text_right])
    assert len(predictions) > 0
    assert type(predictions[0][0]) == np.float32
    shutil.rmtree('.tmpdir')
