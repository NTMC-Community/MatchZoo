import pytest
from matchzoo import engine
from matchzoo.datapack import DataPack


@pytest.fixture
def x():
    data = [['qid0', 'did0', 0],
            ['qid1', 'did1', 1]]
    mapping = {'qid0': [1],
               'qid1': [2],
               'did0': [3],
               'did1': [4]}
    ctx = {'vocab_size': 6, 'fill_word': 6}
    return DataPack(data=data, mapping=mapping, context=ctx)

@pytest.fixture
def test_base_generator(x):
    class MyBaseGenerator(engine.BaseGenerator):
        def __init__(self, inputs, batch_size=1, shuffle=True):
            self.batch_size = batch_size
            super(MyBaseGenerator, self).__init__(batch_size, len(inputs), shuffle)

        def _get_batch_of_transformed_samples(self, index_array):
            batch_x = [0]
            batch_y = [1]
            return (batch_x, batch_y)
    return MyBaseGenerator

def test_base_generator_flow_idnex(x, test_base_generator):
    generator = test_base_generator(x, 4, True)
    for index_array in generator._flow_index():
        x, y = generator._get_batch_of_transformed_samples(index_array)
        assert x == [0]
        assert y == [1]
        break

def test_base_generator_nornal(x, test_base_generator):
    generator = test_base_generator(x, 1, True)
    generator.on_epoch_end()
    x, y = generator[1]
    assert x == [0]
    assert y == [1]
    for index_array in generator._flow_index():
        x, y = generator._get_batch_of_transformed_samples(index_array)
        assert x == [0]
        assert y == [1]
        break

def test_base_generator_except(x, test_base_generator):
    generator = test_base_generator(x, 1, True)
    generator.on_epoch_end()
    with pytest.raises(ValueError):
        x, y = generator[4]
