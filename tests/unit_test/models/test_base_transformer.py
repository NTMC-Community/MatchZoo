import pytest
from matchzoo.engine.base_transformer import TransformerMiXin

def test_transformer():
    class DummyTransformer(TransformerMiXin):
        """This is a dummy class for """

        def __init__(self):
            self.name = 'DummyTransformer'

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            pass

    e = DummyTransformer()
    x = ['a', 'b', 'c']
    y = 0

    # the test
    assert e.name == 'DummyTransformer'

    assert e.fit_transform(x) == None
    assert e.fit_transform(x, y) == None
