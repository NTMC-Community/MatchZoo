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
def cdssm_preprocessor():
    return mz.preprocessors.CDSSMPreprocessor()


@pytest.fixture(scope='module')
def train_data_processed(train_data, cdssm_preprocessor) -> mz.DataPack:
    data = cdssm_preprocessor.fit_transform(train_data)
    return data


@pytest.fixture(scope='module')
def test_data_processed(test_data, cdssm_preprocessor) -> mz.DataPack:
    return cdssm_preprocessor.transform(test_data)


@pytest.fixture(scope='module')
def train_generator(request, train_data_processed):
    return mz.DataGenerator(train_data_processed)


@pytest.fixture(scope='module')
def test_generator(request, test_data_processed):
    return mz.DataGenerator(test_data_processed)


@pytest.mark.slow
def test_cdssm(task,
               train_generator,
               test_generator,
               cdssm_preprocessor):
    """Test CDSSM model."""
    # Create a cdssm model
    cdssm_model = mz.models.CDSSM()
    assert isinstance(cdssm_model.get_default_preprocessor(),
                      mz.preprocessors.CDSSMPreprocessor)
    input_shapes = cdssm_preprocessor.context['input_shapes']
    cdssm_model.params['input_shapes'] = input_shapes
    cdssm_model.params['task'] = task
    cdssm_model.guess_and_fill_missing_params()
    cdssm_model.build()
    cdssm_model.compile()
    cdssm_model.fit_generator(train_generator)
    cdssm_model.save('.tmpdir')

    X, y = test_generator[0]
    try:
        cdssm_model = mz.load_model('.tmpdir')
        predictions = cdssm_model.predict(X)
        assert len(predictions) > 0
        assert type(predictions[0][0]) == np.float32
    finally:
        shutil.rmtree('.tmpdir')
