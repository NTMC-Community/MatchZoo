import pytest
import shutil

import matchzoo as mz
from matchzoo.engine.base_preprocessor import BasePreprocessor


@pytest.fixture
def base_preprocessor():
    BasePreprocessor.__abstractmethods__ = set()
    base_processor = BasePreprocessor()
    return base_processor


def test_save_load(base_preprocessor):
    dirpath = '.tmpdir'
    base_preprocessor.save(dirpath)
    assert mz.load_preprocessor(dirpath)
    with pytest.raises(FileExistsError):
        base_preprocessor.save(dirpath)
    shutil.rmtree(dirpath)
