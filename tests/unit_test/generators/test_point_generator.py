import pytest
from matchzoo import tasks
from matchzoo.generators import PointGenerator
from matchzoo.datapack import DataPack

@pytest.fixture
def x():
    data = [
        {'text_left':[1,2,3], 'text_right': [1,2], 'label': 0, \
        'id': ('did-0', 'did-1')},
        {'text_left':[2,3,4], 'text_right': [3,5], 'label': 1, \
        'id': ('did-2', 'did-3')}
    ]
    cts = {'vocab_size': 6, 'fill_word': 6}
    return DataPack(data, cts)

@pytest.fixture(scope='module', params=[
    tasks.Classification(num_classes=2),
    tasks.Ranking()
])
def task(request):
    return request.param

@pytest.fixture
def raw_point_generator(x, task):
    shuffle = True
    batch_size = 1
    generator = PointGenerator(x, task, batch_size, shuffle)
    return generator

def test_point_generator(raw_point_generator):
    x, y = raw_point_generator[0]
    assert x is not None
    assert y is not None
