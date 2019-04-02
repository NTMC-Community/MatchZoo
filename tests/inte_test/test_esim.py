import shutil

import pytest
import numpy as np
import pandas as pd
import matchzoo as mz
from keras.utils import to_categorical


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
    return mz.tasks.Classification(num_classes=2)


@pytest.fixture(scope='module')
def preprocessor():
    return mz.preprocessors.BasicPreprocessor()


@pytest.fixture(scope='module')
def train_data_processed(train_data, preprocessor) -> mz.DataPack:
    X, Y = preprocessor.fit_transform(train_data).unpack()
    Y = to_categorical(Y)
    df = pd.DataFrame(data = {  'id_left': list(X['id_left']), 
                                'text_left': list(X['text_left']), 
                                'id_right': list(X['id_right']), 
                                'text_right': list(X['text_right']), 
                                'label': list(Y) })
    return mz.pack(df)


@pytest.fixture(scope='module')
def valid_data_processed(valid_data, preprocessor) -> mz.DataPack:
    X, Y = preprocessor.transform(valid_data).unpack()
    Y = to_categorical(Y)
    df = pd.DataFrame(data = {  'id_left': list(X['id_left']), 
                                'text_left': list(X['text_left']), 
                                'id_right': list(X['id_right']), 
                                'text_right': list(X['text_right']), 
                                'label': list(Y) })
    return mz.pack(df)


@pytest.fixture(scope='module')
def test_data_processed(test_data, preprocessor) -> mz.DataPack:
    X, Y = preprocessor.transform(test_data).unpack()
    Y = to_categorical(Y)
    df = pd.DataFrame(data = {  'id_left': list(X['id_left']), 
                                'text_left': list(X['text_left']), 
                                'id_right': list(X['id_right']), 
                                'text_right': list(X['text_right']), 
                                'label': list(Y) })
    return mz.pack(df)


@pytest.fixture(scope='module')
def train_generator(request, train_data_processed):
    return mz.DataGenerator(train_data_processed)


@pytest.mark.slow
def test_duet(train_data_processed,
                 task,
                 train_generator,
                 valid_data_processed,
                 test_data_processed,
                 preprocessor):
    """Test esim model."""
    # Create a esim model
    esim = mz.models.ESIM()
    input_shapes = preprocessor.context['input_shapes']
    embed_dimension = preprocessor.context['vocab_size'] + 1
    esim.params['input_shapes'] = input_shapes
    esim.params['task'] = task
    esim.params['embedding_input_dim'] = embed_dimension
    esim.params['embedding_output_dim'] = 300
    esim.params['lstm_dim'] =  300
    esim.params['mlp_num_units'] = 300
    esim.params['mlp_num_layers'] = 0
    esim.params['mlp_num_fan_out'] = 300
    esim.params['mlp_activation_func'] = 'relu'
    esim.params['dropout_rate'] = 0.5
    esim.guess_and_fill_missing_params()
    esim.build()
    esim.compile()
    
    x_valid, y_valid = valid_data_processed.unpack()
    valid_eval = mz.callbacks.EvaluateAllMetrics(esim,
                                                 x_valid,
                                                 y_valid)
    esim.fit_generator(train_generator, epochs=1, callbacks=[valid_eval])
    esim.save('.tmpdir')
    
    try:
        esim = mz.load_model('.tmpdir')
        x, y = test_data_processed.unpack()
        results = esim.evaluate(x, y)
        assert len(results) > 0
    finally:
        shutil.rmtree('.tmpdir')
