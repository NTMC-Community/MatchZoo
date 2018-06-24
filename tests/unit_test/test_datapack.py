from matchzoo.datapack import DataPack, load_datapack

import pytest
import shutil

@pytest.fixture
def data_pack():
    data = [([1,3], [2,3]), ([3,0], [1,6])]
    ctx = {'vocab_size': 2000}
    return DataPack(data, ctx)

def test_property_size(data_pack):
    assert data_pack.size == 2

def test_unpack(data_pack):
    data, ctx = data_pack.unpack()
    assert not data.empty
    assert isinstance(ctx, dict)

def test_sample(data_pack):
    sampled_data_pack = data_pack.sample(1)
    data, ctx = sampled_data_pack.unpack()
    assert not data.empty
    assert data.shape[0] == 1
    assert isinstance(ctx, dict)

def test_append(data_pack):
    data_pack.append(data_pack)
    data, ctx = data_pack.unpack()
    assert data.shape[0] == 4
    assert isinstance(ctx, dict)

def test_save_load(data_pack):
    dirpath = '.tmpdir'
    data_pack.save(dirpath)
    dp = load_datapack(dirpath)
    with pytest.raises(FileExistsError):
        data_pack.save(dirpath)
    data, ctx = dp.unpack()
    assert data.shape[0] == 2
    assert isinstance(ctx, dict)
    shutil.rmtree(dirpath)
