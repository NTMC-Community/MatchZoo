import pytest
from matchzoo import tasks
from matchzoo.generators import PointGenerator
from matchzoo.datapack import DataPack

@pytest.fixture
def x():
    relation = [['qid0', 'did0', 0],
            ['qid1', 'did1', 1],
            ['qid1', 'did0', 2]]
    content = {'qid0': [1, 2],
               'qid1': [2, 3],
               'did0': [2, 3, 4],
               'did1': [3, 4, 5]}
    ctx = {'vocab_size': 6, 'fill_word': 6}
    columns = ['id_left', 'id_right', 'label']
    return DataPack(relation=relation,
                    content=content,
                    context=ctx,
                    columns=columns
                    )

@pytest.fixture(scope='module', params=[
    tasks.Classification(num_classes=3),
    tasks.Ranking(),
])
def task(request):
    return request.param

@pytest.fixture(scope='module', params=['train', 'test'])
def stage(request):
    return request.param

def test_point_generator(x, task, stage):
    shuffle = False
    batch_size = 3
    generator = PointGenerator(x, task, batch_size, stage, shuffle)
    assert len(generator) == 1
    for x, y in generator:
        assert x['id_left'].tolist() == [[1, 2], [2, 3], [2, 3]]
        assert x['id_right'].tolist() == [[2, 3, 4], [3, 4, 5], [2, 3, 4]]
        assert x['ids'].tolist() == [['qid0', 'did0'],
                            ['qid1', 'did1'],
                            ['qid1', 'did0']]
        if stage == 'test':
            assert y is None
        elif stage == 'train' and task == tasks.Classification(num_classes=3):
            assert y.tolist() == [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
        break

def test_task_mode_in_pointgenerator(x, task, stage):
    generator = PointGenerator(x, task, 1, stage, False)
    assert len(generator) == 3
    with pytest.raises(ValueError):
        x, y = generator[3]

def test_stage_mode_in_pointgenerator(x, task):
    generator = PointGenerator(x, None, 1, 'train', False)
    with pytest.raises(ValueError):
        x, y = generator[0]

