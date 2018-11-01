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
def dssm_preprocessor():
    return preprocessor.DSSMPreprocessor()


@pytest.fixture
def processed_train(train, dssm_preprocessor) -> datapack.DataPack:
    preprocessed_train = dssm_preprocessor.fit_transform(train, stage='train')
    dssm_preprocessor.save('.tmpdir')
    return preprocessed_train


@pytest.fixture
def processed_test(test) -> datapack.DataPack:
    dssm_proprecessor = engine.load_preprocessor('.tmpdir')
    return dssm_proprecessor.fit_transform(test, stage='predict')


@pytest.fixture(params=['point'])
def train_generator(request, processed_train, task) -> engine.BaseGenerator:
    if request.param == 'point':
        return generators.PointGenerator(processed_train,
                                         task=task,
                                         stage='train',
                                         use_word_hashing=True)
    elif request.param == 'pair':
        return generators.PairGenerator(processed_train,
                                        stage='train')


@pytest.fixture(params=['point'])
def test_generator(request, processed_test, task) -> engine.BaseGenerator:
    if request.param == 'point':
        return generators.PointGenerator(processed_test, task=task,
                                         stage='predict',
                                         use_word_hashing=True)
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
