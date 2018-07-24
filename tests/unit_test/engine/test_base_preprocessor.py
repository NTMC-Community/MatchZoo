import pytest
from matchzoo import engine
from matchzoo import preprocessor


def test_base_preprocessor():
    engine.BasePreprocessor.__abstractmethods__ = set()
    base_preprocessor = engine.BasePreprocessor()
    input_tokens = ['animal', 'zoo']
    state, data = base_preprocessor.handle(
        process_unit = preprocessor.VocabularyUnit(),
        input = input_tokens
    )
    assert state.get('term_index')
