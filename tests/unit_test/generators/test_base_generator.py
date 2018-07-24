import pytest
from matchzoo import engine


def test_base_generator():
    class MyBaseGenerator(engine.BaseGenerator):
        def __init__(self, batch_size=10):
            self.batch_size = batch_size
            super(MyBaseGenerator, self).__init__(batch_size, False, False)

        def _total_num_instances(self):
            return 10

        def _get_batches_of_transformed_sample(self, index_array):
            batch_size = len(index_array)
            batch_x = [0]
            batch_y = [1]
            return (batch_x, batch_y)

    generator = MyBaseGenerator()
    assert generator
