import pytest
from matchzoo import engine


def test_base_preprocessor():
    engine.BasePreprocessor.__abstractmethods__ = set()
    base_processor = engine.BasePreprocessor()
    with pytest.raises(AttributeError):
    	base_processor.fit_transform()


