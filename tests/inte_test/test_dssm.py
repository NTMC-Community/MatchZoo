import os
import shutil

import numpy as np
import pytest

import matchzoo as mz


@pytest.fixture(scope='module')
def train_data():
    return mz.datasets.toy.load_data()


@pytest.fixture(scope='module')
def test_data():
    return mz.datasets.toy.load_data(stage='test')


@pytest.fixture(scope='module')
def task(request) -> mz.engine.BaseTask:
    return mz.tasks.Ranking()


@pytest.fixture(scope='module')
def dssm_preprocessor():
    return mz.preprocessors.DSSMPreprocessor()


@pytest.fixture(scope='module')
def train_data_processed(train_data, dssm_preprocessor) -> mz.DataPack:
    data = dssm_preprocessor.fit_transform(train_data)
    return data


@pytest.fixture(scope='module')
def test_data_processed(test_data, dssm_preprocessor) -> mz.DataPack:
    return dssm_preprocessor.transform(test_data)


@pytest.fixture(scope='module')
def train_generator(request, train_data_processed):
    return mz.DataGenerator(train_data_processed)


@pytest.fixture(scope='module')
def test_generator(request, test_data_processed):
    return mz.DataGenerator(test_data_processed)


@pytest.mark.slow
def test_dssm(train_data_processed,
              task,
              train_generator,
              test_generator,
              dssm_preprocessor):
    """Test DSSM model."""
    # Create a dssm model
    dssm_model = mz.models.DSSM()
    input_shapes = dssm_preprocessor.context['input_shapes']
    dssm_model.params['input_shapes'] = input_shapes
    dssm_model.params['task'] = task
    dssm_model.guess_and_fill_missing_params()
    dssm_model.build()
    dssm_model.compile()
    dssm_model.fit_generator(train_generator)
    dssm_model.save('.tmpdir')

    X, y = test_generator[0]
    try:
        dssm_model = mz.load_model('.tmpdir')
        predictions = dssm_model.predict(X)
        assert len(predictions) > 0
        assert type(predictions[0][0]) == np.float32
    finally:
        shutil.rmtree('.tmpdir')
