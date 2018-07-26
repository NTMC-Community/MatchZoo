import pytest
from matchzoo import engine
from matchzoo.datapack import DataPack

@pytest.fixture
def x():
    data = [
        ([1,2,3], [2,3,4], 0, 'did-0', 'did-1'),
        ([3,4,6], [1,3,5], 1, 'did-2', 'did-3')
    ]
    cts = {'vocab_size': 6, 'fill_word': 6}
    return DataPack(data, cts)

def x():
    data = {
        'text_left':[[1,2,3], [2,3,4]],
        'text_right': [[], []],
        'label': [0, 1],
        'id': [('did-0', 'did-1'), ('did-2', 'did-3')]
    }
    cts = {'vocab_size': 6, 'fill_word': 6}
    return DataPack(data, cts)

@pytest.fixture(scope='module', params=[
    tasks.Classification(num_classes=2),
    tasks.Classification(num_classes=16),
    tasks.Ranking()
])
def task(request):
    return request.param

@pytest.fixture
def raw_point_generator(x, task):
    shuffle = True
    batch_size = 1
    generator = PointGenerator(x, task, batch_size, shuffle)
    assert generator

def test_point_generator(generator):
    x, y = generator[0]
    assert x is not None
    assert y is not None
