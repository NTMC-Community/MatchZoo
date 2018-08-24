from matchzoo.datapack import DataPack, load_datapack

import pytest
import shutil

import numpy as np

@pytest.fixture
def data_pack():
    data = [['qid0', 'did0', 1], ['qid1', 'did1', 0]]
    mapping = {'qid0': [1, 2], 'qid1': [2, 3],
               'did0': [2, 3, 4], 'did1': [3, 4, 5]}
    ctx = {'vocab_size': 2000}
    return DataPack(data=data, mapping=mapping, context=ctx)

def test_length(data_pack):
    num_examples = 2
    assert len(data_pack) == num_examples

def test_mapping(data_pack):
    assert data_pack.mapping['qid0'] == [1, 2]

def test_save_load(data_pack):
    dirpath = '.tmpdir'
    data_pack.save(dirpath)
    dp = load_datapack(dirpath)
    with pytest.raises(FileExistsError):
        data_pack.save(dirpath)
    assert len(data_pack) == 2
    shutil.rmtree(dirpath)
