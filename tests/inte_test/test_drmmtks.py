import os
import shutil

import numpy as np
import pytest

import matchzoo as mz


@pytest.fixture(scope='module')
def train_data():
    return mz.datasets.toy.load_train_rank_data()


@pytest.fixture(scope='module')
def test_data():
    return mz.datasets.toy.load_test_rank_data()


@pytest.fixture(scope='module')
def task(request) -> mz.engine.BaseTask:
    return mz.tasks.Ranking()


@pytest.fixture(scope='module')
def drmmtks_preprocessor():
    return mz.preprocessors.DRMMTKSPreprocessor()


@pytest.fixture(scope='module')
def train_data_processed(train_data, drmmtks_preprocessor) -> mz.DataPack:
    data = drmmtks_preprocessor.fit_transform(train_data)
    return data


@pytest.fixture(scope='module')
def test_data_processed(test_data, drmmtks_preprocessor) -> mz.DataPack:
    return drmmtks_preprocessor.transform(test_data)


@pytest.fixture(scope='module')
def train_generator(request, train_data_processed):
    return mz.DataGenerator(train_data_processed)


@pytest.fixture(scope='module')
def test_generator(request, test_data_processed):
    return mz.DataGenerator(test_data_processed)


@pytest.mark.slow
def test_drmmtks(train_data_processed,
              task,
              train_generator,
              test_generator,
              drmmtks_preprocessor):
    """Test DRMMTKS model."""
    # Create a drmmtks model
    drmmtks_model = mz.models.DRMMTKSModel()
    input_shapes = drmmtks_preprocessor.context['input_shapes']
    vocab_size = drmmtks_preprocessor.context['vocab_size']
    drmmtks_model.params['input_shapes'] = input_shapes
    drmmtks_model.params['task'] = task
    drmmtks_model.params['vocab_size'] = vocab_size
    drmmtks_model.guess_and_fill_missing_params()
    drmmtks_model.build()
    drmmtks_model.compile()
    drmmtks_model.fit_generator(train_generator)
    drmmtks_model.save('.tmpdir')

    X, y = test_generator[0]
    try:
        drmmtks_model = mz.load_model('.tmpdir')
        predictions = drmmtks_model.predict(X, y)
        assert len(predictions) > 0
        assert type(predictions[0][0]) == np.float32
    finally:
        shutil.rmtree('.tmpdir')
