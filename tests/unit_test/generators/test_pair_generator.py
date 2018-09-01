import pytest
import pandas as pd
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
    right = pd.DataFrame(right, columns=['id_right', 'text_right'])
    right.set_index('id_right', inplace=True)
    return DataPack(relation=relation,
                    left=left,
                    right=right,
                    context=ctx
                    )

def test_pair_generator_one(x):
    """Test pair generator with only one negative sample."""
    shuffle = False
    batch_size = 1
    generator = PairGenerator(inputs=x,
                              num_neg=1,
                              dup_time=2,
                              batch_size=batch_size,
                              stage='train',
                              shuffle=shuffle)
    assert len(generator) == 4
    for idx, (x, y) in enumerate(generator):
        if idx == len(generator):
            break
        assert x is not None
        assert y is not None

def test_pair_generator_multi(x):
    """Test pair generator with multiple negative sample."""
    shuffle = False
    batch_size = 1
    generator = PairGenerator(x, 2, 2, batch_size, 'test', shuffle)
    assert len(generator) == 2
    for x, y in generator:
        assert x is not None
        assert y is not None
