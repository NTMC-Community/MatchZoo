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
    with open(os.path.join(path, 'train_rank.txt')) as f:
        train = [tuple(map(str, i.strip().split('\t'))) for i in f]
    return train

@pytest.fixture
def test():
    test = []
    path = os.path.dirname(__file__)
    with open(os.path.join(path, 'test_rank.txt')) as f:
        test = [tuple(map(str, i.strip().split('\t'))) for i in f]
    return test

@pytest.fixture(scope='module', params=[
    tasks.Classification(num_classes=3),
    tasks.Ranking(),
])
def task(request) -> engine.BaseTask:
    return request.param

@pytest.fixture
def dssm_preprocessor():
    return preprocessor.DSSMPreprocessor()

@pytest.fixture(scope='session')
def processed_train(train, dssm_preprocessor) -> datapack.DataPack:
    preprocessed_train = dssm_preprocessor.fit_transform(train, stage='train')
    dssm_preprocessor.save('.tmpdir')
    return preprocessed_train

@pytest.fixture
def processed_test(test) -> datapack.DataPack:
    dssm_proprecessor = engine.load_preprocessor('.tmpdir')
    return dssm_proprecessor.fit_transform(test, stage='test')

#@pytest.fixture
@pytest.fixture(scope='session', params=['point', 'pair'])
def train_generator(request,
                    processed_train,
                    task) -> generators.PointGenerator:
    if request.param == 'point':
        return generators.PointGenerator(processed_train,
                                         task=task,
                                         stage='train')
    elif request.param == 'pair':
        return generators.PairGenerator(processed_train,
                                         task=task,
                                         stage='train')

@pytest.fixture
def test_generator(processed_test, task) -> generators.PointGenerator:
    return generators.PointGenerator(processed_test, task=task, stage='test')

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

'''
def test_pair_dssm(train, test):
    """Test DSSM model."""
    # do pre-processing.
    dssm_preprocessor = preprocessor.DSSMPreprocessor()
    processed_train = dssm_preprocessor.fit_transform(train, stage='train')
    processed_test = dssm_preprocessor.fit_transform(test, stage='test')
    # the dimension of dssm model is the length of tri-letters.
    task = tasks.Ranking()
    # generator.
    generator = generators.PairGenerator(processed_train, stage='train')
    # Create a dssm model
    dssm_model = models.DSSMModel()
    dssm_model.params['input_shapes'] = processed_train.context['input_shapes']
    dssm_model.params['task'] = task
    dssm_model.guess_and_fill_missing_params()
    dssm_model.build()
    dssm_model.compile()
    dssm_model.fit_generator(generator)
    # save
    dssm_preprocessor.save('.tmpdir')
    dssm_model.save('.tmpdir')

    # testing
    dssm_proprecessor = engine.load_preprocessor('.tmpdir')
    processed_test = dssm_proprecessor.fit_transform(test, stage='test')
    generator = generators.PointGenerator(processed_test, task=task, stage='test')
    X, y = generator[0]
    dssm_model = engine.load_model('.tmpdir')
    predictions = dssm_model.predict([X.text_left, X.text_right])
    assert len(predictions) > 0
    assert type(predictions[0][0]) == np.float32
    shutil.rmtree('.tmpdir')
'''
