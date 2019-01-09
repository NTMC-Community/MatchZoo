import shutil

import numpy as np
import pytest

import matchzoo as mz


@pytest.fixture(scope='module')
def train_data():
    return mz.datasets.toy.load_data(stage='train')


@pytest.fixture(scope='module')
def valid_data():
    return mz.datasets.toy.load_data(stage='dev')


@pytest.fixture(scope='module')
def test_data():
    return mz.datasets.toy.load_data(stage='test')


@pytest.fixture(scope='module')
def task(request) -> mz.engine.BaseTask:
    return mz.tasks.Ranking()


@pytest.fixture(scope='module')
def preprocessor():
    return mz.preprocessors.BasicPreprocessor()


@pytest.fixture(scope='module')
def train_data_processed(train_data, preprocessor) -> mz.DataPack:
    data = preprocessor.fit_transform(train_data)
    return data


@pytest.fixture(scope='module')
def valid_data_processed(valid_data, preprocessor) -> mz.DataPack:
    return preprocessor.transform(valid_data)


@pytest.fixture(scope='module')
def test_data_processed(test_data, preprocessor) -> mz.DataPack:
    return preprocessor.transform(test_data)


@pytest.fixture(scope='module')
def train_generator(request, train_data_processed):
    return mz.DataGenerator(train_data_processed)


@pytest.mark.slow
def test_mvlstm(train_data_processed,
                 task,
                 train_generator,
                 valid_data_processed,
                 test_data_processed,
                 preprocessor):
    """Test MVLSTM model."""
    # Create a mvlstm model
    mvlstm = mz.models.MVLSTM()
    input_shapes = preprocessor.context['input_shapes']
    embed_dimension = preprocessor.context['vocab_size'] + 1
    mvlstm.params['input_shapes'] = input_shapes
    mvlstm.params['task'] = task
    mvlstm.params['embedding_input_dim'] = embed_dimension
    mvlstm.params['embedding_output_dim'] = 10
    mvlstm.params['lstm_units'] = 10
    mvlstm.params['top_k'] = 10
    mvlstm.params['mlp_num_layers'] = 1
    mvlstm.params['mlp_num_units'] = 5
    mvlstm.params['mlp_num_fan_out'] = 1
    mvlstm.params['mlp_activation_func'] = 'relu'
    mvlstm.params['dropout_rate'] = 0.5
    mvlstm.guess_and_fill_missing_params()
    mvlstm.build()
    mvlstm.compile()

    x_valid, y_valid = valid_data_processed.unpack()
    valid_eval = mz.callbacks.EvaluateAllMetrics(mvlstm,
                                                 x_valid,
                                                 y_valid)
    mvlstm.fit_generator(train_generator, epochs=1,
                                callbacks=[valid_eval])
    mvlstm.save('.tmpdir')

    try:
        mvlstm = mz.load_model('.tmpdir')
        x, y = test_data_processed.unpack()
        results = mvlstm.evaluate(x, y)
        assert len(results) > 0
    finally:
        shutil.rmtree('.tmpdir')
