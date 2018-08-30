import pytest
import pandas as pd
from matchzoo import engine
from matchzoo.datapack import DataPack


@pytest.fixture
def x():
    relation = [['qid0', 'did0', 0], ['qid1', 'did1', 1]]
    left = [['qid0', [1]], ['qid1', [2]]]
    right = [['did0', [3]], ['did1', [4]]]
    ctx = {'vocab_size': 6, 'fill_word': 6}
    relation = pd.DataFrame(relation, columns=['id_left', 'id_right', 'label'])
    left = pd.DataFrame(left, columns=['id_left', 'text_left'])
    left.set_index('id_left', inplace=True)
    right = pd.DataFrame(right, columns=['id_right', 'text_right'])
    right.set_index('id_right', inplace=True)
    return DataPack(relation=relation,
                    left=left,
                    right=right,
                    context=ctx)

@pytest.fixture(scope='module', params=['train', 'test'])
def stage(request):
    return request.param

@pytest.fixture(scope='module', params=[1, 2, 3, 4])
def batch_size(request):
    return request.param

@pytest.fixture(scope='module', params=[True, False])
def shuffle(request):
    return request.param

@pytest.fixture
def generator(x, stage, batch_size, shuffle):
    class MyBaseGenerator(engine.BaseGenerator):
        def __init__(self, inputs=x, batch_size=1, stage=stage, shuffle=True):
            self.batch_size = batch_size
            super(MyBaseGenerator, self).__init__(batch_size, len(inputs),
                                                  stage, shuffle)

        def _get_batch_of_transformed_samples(self, index_array):
            batch_x = [0]
            batch_y = [1]
            return (batch_x, batch_y)
    return MyBaseGenerator()

def test_base_generator_train(generator):
    assert len(generator) == 2
    generator.reset()
    for idx, (x, y) in enumerate(generator):
        assert x == [0]
        assert y == [1]
        if idx > len(generator):
            break

def test_base_generator_except(generator):
    generator.on_epoch_end()
    with pytest.raises(ValueError):
        x, y = generator[4]
