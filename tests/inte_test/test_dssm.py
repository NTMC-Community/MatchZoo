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
    print(train[:2])
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
def task(request):
    return request.param

def test_point_dssm(train, test, task):
    """Test DSSM model."""
    # do pre-processing.
    dssm_preprocessor = preprocessor.DSSMPreprocessor()
    processed_train = dssm_preprocessor.fit_transform(train, stage='train')
    processed_test = dssm_preprocessor.fit_transform(test, stage='test')
    # the dimension of dssm model is the length of tri-letters.
    input_shapes = processed_train.context['input_shapes']
    # generator.
    generator = generators.PointGenerator(processed_train, task=task, stage='train')
    # Create a dssm model
    dssm_model = models.DSSMModel()
    dssm_model.params['input_shapes'] = input_shapes
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
    generator = generators.PointGenerator(processed_test, stage='test')
    X, y = generator[0]
    dssm_model = engine.load_model('.tmpdir')
    predictions = dssm_model.predict([X.text_left, X.text_right])
    assert len(predictions) > 0
    assert type(predictions[0][0]) == np.float32
    shutil.rmtree('.tmpdir')

def test_pair_dssm(train, test):
    """Test DSSM model."""
    # do pre-processing.
    dssm_preprocessor = preprocessor.DSSMPreprocessor()
    processed_train = dssm_preprocessor.fit_transform(train, stage='train')
    processed_test = dssm_preprocessor.fit_transform(test, stage='test')
    # the dimension of dssm model is the length of tri-letters.
    input_shapes = processed_train.context['input_shapes']
    task = tasks.Ranking()
    # generator.
    generator = generators.PairGenerator(processed_train, stage='train')
    # Create a dssm model
    dssm_model = models.DSSMModel()
    dssm_model.params['input_shapes'] = input_shapes
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
