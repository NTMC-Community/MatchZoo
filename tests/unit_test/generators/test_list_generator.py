import pytest
import pandas as pd
import numpy as np
from matchzoo.generators import ListGenerator
from matchzoo.datapack import DataPack

@pytest.fixture
def x():
    relation = [['qid0', 'did0', 0],
            ['qid1', 'did1', 1],
            ['qid1', 'did0', 2]]
    left = [['qid0', [1, 2]],
                 ['qid1', [2, 3]]]
    right = [['did0', [2, 3, 4]],
                  ['did1', [3, 4, 5]]]
    ctx = {'vocab_size': 6, 'fill_word': 6}
    relation = pd.DataFrame(relation, columns=['id_left', 'id_right', 'label'])
    left = pd.DataFrame(left, columns=['id_left', 'text_left'])
    left.set_index('id_left', inplace=True)
    right = pd.DataFrame(right, columns=['id_right', 'text_right'])
    right.set_index('id_right', inplace=True)
    return DataPack(relation=relation,
                    left=left,
                    right=right,
                    context=ctx
                    )

@pytest.fixture(scope='module', params=['train', 'test'])
def stage(request):
    return request.param

def test_list_generator(x, stage):
    """Test list generator"""
    np.random.seed(111)
    shuffle = False
    generator = ListGenerator(x, 1, stage, shuffle)
    assert len(generator) == 2
    x0, y0 = generator[0]
    x1, y1 = generator[1]
    assert x0['ids'].tolist() == [['qid0', 'did0']]
    assert x0['text_left'].tolist() == [[1, 2]]
    assert x0['text_right'].tolist() == [[2, 3, 4]]
    assert x1['ids'].tolist() == [['qid1', 'did1'], ['qid1', 'did0']]
    assert x1['text_left'].tolist() == [[2, 3], [2, 3]]
    assert x1['text_right'].tolist() == [[3, 4, 5], [2, 3, 4]]
    if stage == 'test':
        assert y0 is None
        assert y1 is None
    elif stage == 'train':
        assert y0.tolist() == [0]
        assert y1.tolist() == [1, 2]
