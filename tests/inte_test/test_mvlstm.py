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
def mvlstm_preprocessor():
    return mz.preprocessors.DSSMPreprocessor()


@pytest.fixture(scope='module')
def train_data_processed(train_data, mvlstm_preprocessor) -> mz.DataPack:
    data = mvlstm_preprocessor.fit_transform(train_data)
    return data


@pytest.fixture(scope='module')
def test_data_processed(test_data, mvlstm_preprocessor) -> mz.DataPack:
    return mvlstm_preprocessor.transform(test_data)


@pytest.fixture(scope='module')
def train_generator(request, train_data_processed):
    return mz.DataGenerator(train_data_processed)


@pytest.fixture(scope='module')
def test_generator(request, test_data_processed):
    return mz.DataGenerator(test_data_processed)


@pytest.mark.slow
def test_mvlstm(train_data_processed,
              task,
              train_generator,
              test_generator,
              mvlstm_preprocessor):
    """Test DSSM model."""
    # Create a mvlstm model
    mvlstm_model = mz.models.DSSMModel()
    input_shapes = mvlstm_preprocessor.context['input_shapes']
    mvlstm_model.params['input_shapes'] = input_shapes
    mvlstm_model.params['task'] = task
    mvlstm_model.guess_and_fill_missing_params()
    mvlstm_model.build()
    mvlstm_model.compile()
    mvlstm_model.fit_generator(train_generator)
    mvlstm_model.save('.tmpdir')

    X, y = test_generator[0]
    try:
        mvlstm_model = mz.load_model('.tmpdir')
        predictions = mvlstm_model.predict(X)
        assert len(predictions) > 0
        assert type(predictions[0][0]) == np.float32
    finally:
        shutil.rmtree('.tmpdir')
