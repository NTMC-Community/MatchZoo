import pytest
import numpy as np
from matchzoo import tasks
from matchzoo.generators import PairGenerator
from matchzoo.datapack import DataPack

@pytest.fixture
def x():
    data = [
        {'text_left':[1], 'text_right': [2], \
         'id_left': 'qid0', 'id_right': 'did1', \
        'label': 0},
        {'text_left':[1], 'text_right': [3], \
         'id_left': 'qid0', 'id_right': 'did2', \
         'label': 1},
        {'text_left':[1], 'text_right': [4], \
         'id_left': 'qid0', 'id_right': 'did3', \
         'label': 2}
    ]
    cts = {'vocab_size': 6, 'fill_word': 6}
    columns = ['text_left', 'text_right', 'id_left', 'id_right', 'label']
    return DataPack(data, cts, columns)

@pytest.fixture
def task():
    return tasks.Ranking()

def test_pair_generator_one(x, task):
    """Test pair generator with only one negative sample."""
    shuffle = False
    batch_size = 1
    generator = PairGenerator(x, 1, task, batch_size, shuffle)
    x, y = generator[0]
    assert np.array_equal(x.text_left, np.array([[1], [1]]))
    assert np.array_equal(x.text_right, np.array([[4], [3]]))
    assert np.array_equal(x.id_left, np.array(['qid0', 'qid0']))
    assert np.array_equal(x.id_right, np.array(['did3', 'did2']))
    assert y == [2, 1]

def test_pair_generator_multi(x, task):
    """Test pair generator with multiple negative sample."""
    shuffle = False
    batch_size = 1
    generator = PairGenerator(x, 2, task, batch_size, shuffle)
    x, y = generator[0]
    assert np.array_equal(x.text_left, np.array([[1], [1], [1]]))
    assert np.array_equal(x.text_right, np.array([[4], [3], [2]]))
    assert np.array_equal(x.id_left, np.array(['qid0', 'qid0', 'qid0']))
    assert np.array_equal(x.id_right, np.array(['did3', 'did2', 'did1']))
    assert y == [2, 1, 0]


