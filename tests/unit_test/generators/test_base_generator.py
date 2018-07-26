import pytest
from matchzoo import engine
from matchzoo.datapack import DataPack


@pytest.fixture
def x():
    data = [
        ([1,2,3], [2,3,4], 0, 'did-0', 'did-1'),
        ([3,4,6], [1,3,5], 1, 'did-2', 'did-3')
    ]
    cts = {'vocab_size': 6, 'fill_word': 6}
    return DataPack(data, cts)

def test_base_generator(x):
    class MyBaseGenerator(engine.BaseGenerator):
        def __init__(self, inputs, batch_size=1, shuffle=True):
            self.batch_size = batch_size
            self.n = len(inputs)
            super(MyBaseGenerator, self).__init__(batch_size, shuffle)

        def _get_batch_of_transformed_samples(self, index_array):
            batch_x = [0]
            batch_y = [1]
            return (batch_x, batch_y)

    generator = MyBaseGenerator(x)
    x, y = generator[0]
    assert generator
    generator.on_epoch_end()
    x, y = generator[1]
    assert x == [0]
    assert y == [1]
    for index_array in generator._flow_index():
        x, y = generator._get_batch_of_transformed_samples(index_array)
        assert x == [0]
        assert y == [1]
