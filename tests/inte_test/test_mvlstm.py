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
    return mz.preprocessors.MVLSTMPreprocessor()


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
    """Test MVLSTM model."""
    # Create a mvlstm model
    mvlstm_model = mz.models.MVLSTMModel()
    input_shapes = mvlstm_preprocessor.context['input_shapes']
    term_index = mvlstm_preprocessor.context['vocab_unit'].state['term_index']
    embedding_input_dim = mvlstm_preprocessor.context['embedding_input_dim']
    mvlstm_model.params['input_shapes'] = input_shapes
    mvlstm_model.params['task'] = task
    mvlstm_model.params['embedding_input_dim'] = embedding_input_dim

    parent_path = os.path.dirname(__file__)
    parent_path = os.path.dirname(parent_path)
    parent_path = os.path.dirname(parent_path) 
    #parent_path = os.path.dirname(parent_path) 
    embed_path = os.path.join(
                    parent_path,
                    'matchzoo/datasets/embeddings/embed_rank.txt')
    embedding = mz.embedding.load_from_file(embed_path)
    embedding_matrix = embedding.build_matrix(term_index)

    mvlstm_model.guess_and_fill_missing_params()
    mvlstm_model.build()
    mvlstm_model.load_embedding_matrix(embedding_matrix)
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
