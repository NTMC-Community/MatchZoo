import os
import pytest
import shutil
import numpy as np

from matchzoo import datapack
from matchzoo import generators
from matchzoo import preprocessor
from matchzoo import models
from matchzoo import engine

@pytest.fixture
def train():
    train = []
    path = os.path.dirname(__file__)
    with open(os.path.join(path, 'train.txt')) as f:
        train = [tuple(map(str, i.split('\t'))) for i in f]
    return train

@pytest.fixture
def test():
    test = []
    path = os.path.dirname(__file__)
    with open(os.path.join(path, 'test.txt')) as f:
        test = [tuple(map(str, i.split('\t'))) for i in f]
    return test



def test_dssm(train, test):
    """Test DSSM model."""
    # do pre-processing.
    dssm_preprocessor = preprocessor.DSSMPreprocessor()
    processed_train = dssm_preprocessor.fit_transform(train, stage='train')
    # the dimension of dssm model is the length of tri-letters.
    input_shapes = processed_train.context['input_shapes']
    # generator.
    generator = generators.PointGenerator(processed_train)
    # X, y = generator[0]
    # Create a dssm model
    dssm_model = models.DSSMModel()
    dssm_model.params['input_shapes'] = input_shapes
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
    generator = generators.PointGenerator(processed_test)
    X, y = generator[0]
    dssm_model = engine.load_model('.tmpdir')
    predictions = dssm_model.predict([X.text_left, X.text_right])
    assert len(predictions) > 0
    assert type(predictions[0][0]) == np.float32
    shutil.rmtree('.tmpdir')
