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
def arci_preprocessor(request):
    return request.param

@pytest.fixture
def processed_train(train, arci_preprocessor) -> datapack.DataPack:
    preprocessed_train = arci_preprocessor.fit_transform(train, stage='train')
    arci_preprocessor.save('.tmpdir')
    return preprocessed_train

@pytest.fixture
def processed_test(test) -> datapack.DataPack:
    arci_proprecessor = engine.load_preprocessor('.tmpdir')
    return arci_proprecessor.fit_transform(test, stage='predict')

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
        return generators.PointGenerator(processed_test, task=task, stage='predict')
    elif request.param == 'list':
        return generators.ListGenerator(processed_test, stage='predict')


@pytest.mark.slow
def test_arci(processed_train,
              task,
              train_generator,
              test_generator):
    """Test ArcI model."""
    # Create a arci model
    arci_model = models.ArcIModel()
    arci_model.params['input_shapes'] = processed_train.context['input_shapes']
    arci_model.params['vocab_size'] = \
        len(processed_train.context['term_index']) + 1
    if 'embedding_mat' in processed_train.context:
        arci_model.embedding_mat = processed_train.context['embedding_mat']
    arci_model.params['task'] = task
    arci_model.guess_and_fill_missing_params()
    arci_model.build()
    arci_model.compile()
    arci_model.fit_generator(train_generator)
    # save
    arci_model.save('.tmpdir')

    # testing
    X, y = test_generator[0]
    arci_model = engine.load_model('.tmpdir')
    predictions = arci_model.predict([X.text_left, X.text_right])
    assert len(predictions) > 0
    assert type(predictions[0][0]) == np.float32
    shutil.rmtree('.tmpdir')
