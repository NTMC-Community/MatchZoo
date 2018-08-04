import pytest
from matchzoo import tasks
from matchzoo.generators import PointGenerator
from matchzoo.datapack import DataPack

@pytest.fixture
def x():
    data = [
        {'text_left':[1,2,3], 'text_right': [1,2], \
         'id_left': 'did-0', 'id_right': 'did-1', \
        'label': 0},
        {'text_left':[2,3,4], 'text_right': [3,5], \
         'id_left': 'did-2', 'id_right': 'did-3', \
         'label': 1}
    ]
    cts = {'vocab_size': 6, 'fill_word': 6}
    return DataPack(data, cts)

@pytest.fixture(scope='module', params=[
    tasks.Classification(num_classes=2),
    tasks.Ranking(),
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

def test_taskmode_in_pointgenerator(x):
    generator = PointGenerator(x, None, 1, False)
    with pytest.raises(ValueError):
        x, y = generator[0]
