import pytest
import pandas as pd
from matchzoo import tasks
from matchzoo.generators import PointGenerator
from matchzoo.datapack import DataPack

@pytest.fixture
def x():
    relation = [['qid0', 'did0', 0],
            ['qid1', 'did1', 1],
            ['qid1', 'did0', 2]]
    left_data = [['qid0', [1, 2]],
                 ['qid1', [2, 3]]]
    right_data = [['did0', [2, 3, 4]],
                  ['did1', [3, 4, 5]]]
    ctx = {'vocab_size': 6, 'fill_word': 6}
    col_relation = ['id_left', 'id_right', 'label']
    col_left = ['id_left', 'text_left']
    col_right = ['id_right', 'text_right']
    relation_df = pd.DataFrame(relation, columns=col_relation)
    left_df = pd.DataFrame(left_data, columns=col_left)
    left_df.set_index('id_left', inplace=True)
    right_df = pd.DataFrame(right_data, columns=col_right)
    right_df.set_index('id_right', inplace=True)
    return DataPack(relation=relation_df,
                    left_data=left_df,
                    right_data=right_df,
                    context=ctx
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
        assert x['ids'].tolist() == [['qid0', 'did0'],
                            ['qid1', 'did1'],
                            ['qid1', 'did0']]
        assert x['text_left'].tolist() == [[1, 2], [2, 3], [2, 3]]
        assert x['text_right'].tolist() == [[2, 3, 4], [3, 4, 5], [2, 3, 4]]
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
