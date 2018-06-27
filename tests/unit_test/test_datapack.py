from matchzoo.datapack import DataPack, load_datapack

import pytest
import shutil

import numpy as np

@pytest.fixture
def data_pack():
    data = np.zeros((2, 2))
    ctx = {'vocab_size': 2000}
    return DataPack(data, ctx)

def test_length(data_pack):
    num_examples = 2
    assert len(data_pack) == num_examples

def test_sample(data_pack):
    sampled_data_pack = data_pack.sample(1)
    assert len(sampled_data_pack) == 1

def test_append(data_pack):
    data_pack.append(data_pack)
    assert len(data_pack) == 4
    assert data_pack.context == {'vocab_size': 2000}

def test_save_load(data_pack):
    dirpath = '.tmpdir'
    data_pack.save(dirpath)
    dp = load_datapack(dirpath)
    with pytest.raises(FileExistsError):
        data_pack.save(dirpath)
    assert len(data_pack) == 2
    shutil.rmtree(dirpath)
