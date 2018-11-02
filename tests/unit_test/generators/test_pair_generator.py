import pytest
import pandas as pd
import numpy as np
from matchzoo.generators import PairGenerator
from matchzoo.datapack import DataPack

@pytest.fixture
def x():
    relation = [['qid0', 'did0', 0],
                ['qid0', 'did1', 1],
                ['qid0', 'did2', 2]]
    left = [['qid0', [1, 2]]]
    right = [['did0', [1, 2, 3]],
             ['did1', [2, 3, 4]],
             ['did2', [3, 4, 5]]]
    ctx = {'vocab_size': 6, 'fill_word': 6}
    relation = pd.DataFrame(relation, columns=['id_left', 'id_right', 'label'])
    left = pd.DataFrame(left, columns=['id_left', 'text_left'])
    left.set_index('id_left', inplace=True)
    left['length_left'] = left.apply(lambda x: len(x['text_left']), axis=1)
    right = pd.DataFrame(right, columns=['id_right', 'text_right'])
    right.set_index('id_right', inplace=True)
    right['length_right'] = right.apply(lambda x: len(x['text_right']), axis=1)
    return DataPack(relation=relation,
                    left=left,
                    right=right,
                    context=ctx
                    )

def test_pair_generator_one(x):
    """Test pair generator with only one negative sample."""
    np.random.seed(111)
    shuffle = False
    batch_size = 1
    generator = PairGenerator(inputs=x,
                              num_neg=1,
                              num_dup=2,
                              batch_size=batch_size,
                              stage='train',
                              shuffle=shuffle)
    assert len(generator) == 4
    x0, y0 = generator[0]
    assert x0['text_left'].tolist() == [[1, 2], [1, 2]]
    assert x0['text_right'].tolist() == [[3, 4, 5], [1, 2, 3]]
    assert x0['id_left'].tolist() == ['qid0', 'qid0']
    assert x0['id_right'].tolist() == ['did2', 'did0']
    assert x0['length_left'].tolist() == [2, 2]
    assert x0['length_right'].tolist() == [3, 3]
    assert y0.tolist() == [2, 0]

def test_pair_generator_multi(x):
    """Test pair generator with multiple negative sample."""
    np.random.seed(111)
    shuffle = False
    batch_size = 1
    generator = PairGenerator(x, 2, 2, batch_size, 'predict', shuffle)
    assert len(generator) == 2
    x0, y0 = generator[0]
    assert x0['text_left'].tolist() == [[1, 2], [1, 2], [1, 2]]
    assert x0['text_right'].tolist() == [[3, 4, 5], [1, 2, 3], [2, 3, 4]]
    assert x0['id_left'].tolist() == ['qid0', 'qid0', 'qid0']
    assert x0['id_right'].tolist() == ['did2', 'did0', 'did1']
    assert x0['length_left'].tolist() == [2, 2, 2]
    assert x0['length_right'].tolist() == [3, 3, 3]
    assert y0.tolist() == [2, 0, 1]
