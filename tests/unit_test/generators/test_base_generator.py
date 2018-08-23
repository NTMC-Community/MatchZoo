import pytest
from matchzoo import engine
from matchzoo.datapack import DataPack


@pytest.fixture
def x():
    data = [
        ([1,2,3], [2,3,4], 'did-0', 'did-1', 0),
        ([3,4,6], [1,3,5], 'did-2', 'did-3', 1),
        ([1,4,5], [2,4,5], 'did-2', 'did-4', 1),
    ]
    cts = {'vocab_size': 6, 'fill_word': 6}
    return DataPack(data, context=cts)

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
