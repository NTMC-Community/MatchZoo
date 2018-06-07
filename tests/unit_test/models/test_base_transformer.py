import pytest
from matchzoo.engine.base_transformer import BaseTransformer

def test_transformer():
    class DummyTransformer(BaseTransformer):
        """This is a dummy class for """

        def __init__(self):
            self.params['name'] = 'DummyTransformer'
            print(self.params['transformer_class'])

        def build_vocabulary(self, X, fixed_vocab=False):
            return self

        def transform(self, X):
            pass

        def fit_transform(self, X):
            pass

    e = DummyTransformer()
    x = ['a', 'b', 'c']
    y = 0

    # the test
    assert e.params['name'] == 'DummyTransformer'

    assert e.fit_transform(x) == None
    assert e.fit_transform(x, y) == None
