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
    with open(os.path.join(path, '../sample/train_rank.txt')) as f:
        train = [tuple(map(str, i.strip().split('\t'))) for i in f]
    return train


@pytest.fixture
def test():
    test = []
    path = os.path.dirname(__file__)
    with open(os.path.join(path, '../sample/test_rank.txt')) as f:
        test = [tuple(map(str, i.strip().split('\t'))) for i in f]
    return test


@pytest.fixture
def task(request) -> engine.BaseTask:
    return tasks.Ranking()


@pytest.fixture(scope='module', params=[
    preprocessor.ArcIPreprocessor(),
    preprocessor.ArcIPreprocessor(fixed_length=[10, 10],
                                  embedding_file=os.path.join(
                                          os.path.dirname(__file__),
                                          '../sample/embed_rank.txt'
                                  )
                                  )
])
def mvlstm_preprocessor(request):
    return request.param


@pytest.fixture
def processed_train(train, mvlstm_preprocessor) -> datapack.DataPack:
    preprocessed_train = mvlstm_preprocessor.fit_transform(train, stage='train')
    mvlstm_preprocessor.save('.tmpdir')
    return preprocessed_train


@pytest.fixture
def processed_test(test) -> datapack.DataPack:
    mvlstm_proprecessor = engine.load_preprocessor('.tmpdir')
    return mvlstm_proprecessor.fit_transform(test, stage='predict')


@pytest.fixture(params=['point', 'pair'])
def train_generator(request, processed_train, task) -> engine.BaseGenerator:
    if request.param == 'point':
        return generators.PointGenerator(processed_train,
                                         task=task,
                                         stage='train')
    elif request.param == 'pair':
        return generators.PairGenerator(processed_train,
                                        stage='train')


@pytest.fixture(params=['point', 'list'])
def test_generator(request, processed_test, task) -> engine.BaseGenerator:
    if request.param == 'point':
        return generators.PointGenerator(processed_test, task=task,
                                         stage='predict')
    elif request.param == 'list':
        return generators.ListGenerator(processed_test, stage='predict')


@pytest.mark.slow
def test_mvlstm(processed_train,
              task,
              train_generator,
              test_generator):
    """Test ArcI model."""
    # Create a mvlstm model
    mvlstm_model = models.MvLstmModel()
    mvlstm_model.params['input_shapes'] = processed_train.context['input_shapes']
    mvlstm_model.params['vocab_size'] = \
        len(processed_train.context['term_index']) + 1
    if 'embedding_mat' in processed_train.context:
        mvlstm_model.embedding_mat = processed_train.context['embedding_mat']
    mvlstm_model.params['task'] = task
    mvlstm_model.params['top_k'] = 10
    mvlstm_model.guess_and_fill_missing_params()
    mvlstm_model.build()
    mvlstm_model.compile()
    mvlstm_model.fit_generator(train_generator)
    # save
    mvlstm_model.save('.tmpdir')

    # testing
    X, y = test_generator[0]
    mvlstm_model = engine.load_model('.tmpdir')
    predictions = mvlstm_model.predict([X.text_left, X.text_right])
    assert len(predictions) > 0
    assert type(predictions[0][0]) == np.float32
    shutil.rmtree('.tmpdir')
