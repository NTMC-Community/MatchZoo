import pytest
from matchzoo import tasks
from matchzoo.generators import PointGenerator
from matchzoo.datapack import DataPack

@pytest.fixture
def x():
    data = [['qid0', 'did0', 0], ['qid1', 'did1', 1]]
    mapping = {'qid0': [1, 2],
               'qid1': [2, 3],
               'did0': [2, 3, 4],
               'did1': [3, 4, 5]}
    cts = {'vocab_size': 6, 'fill_word': 6}
    columns = ['id_left', 'id_right', 'label']
    return DataPack(data=data,
                    mapping=mapping,
                    context=cts,
                    columns=columns
                    )

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
    generator = PointGenerator(x, task, batch_size, 'train', shuffle)
    return generator

def test_point_generator(raw_point_generator):
    x, y = raw_point_generator[0]
    assert x is not None
    assert y is not None

def test_taskmode_in_pointgenerator(x):
    generator = PointGenerator(x, None, 1, 'train', False)
    with pytest.raises(ValueError):
        x, y = generator[0]
