import pytest
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

def test_pair_generator_one(x):
    """Test pair generator with only one negative sample."""
    shuffle = False
    batch_size = 1
    generator = PairGenerator(x, 1, 1, batch_size, shuffle)
    x, y = generator[0]
    assert x is not None
    assert y is not None

def test_pair_generator_multi(x):
    """Test pair generator with multiple negative sample."""
    shuffle = False
    batch_size = 1
    generator = PairGenerator(x, 2, 2, batch_size, shuffle)
    x, y = generator[0]
    assert x is not None
    assert y is not None

