import pytest
import shutil
from matchzoo import engine


@pytest.fixture
def base_preprocessor():
    engine.BasePreprocessor.__abstractmethods__ = set()
    base_processor = engine.BasePreprocessor()
    return base_processor


def test_save_load(base_preprocessor):
    dirpath = '.tmpdir'
    base_preprocessor.save(dirpath)
    assert engine.load_preprocessor(dirpath)
    with pytest.raises(FileExistsError):
        base_preprocessor.save(dirpath)
    shutil.rmtree(dirpath)
