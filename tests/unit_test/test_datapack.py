from matchzoo.datapack import DataPack, load_datapack

import pytest
import shutil

import numpy as np

@pytest.fixture
def data_pack():
    relation = [['qid0', 'did0', 1], ['qid1', 'did1', 0]]
    content = {'qid0': [1, 2], 'qid1': [2, 3],
               'did0': [2, 3, 4], 'did1': [3, 4, 5]}
    ctx = {'vocab_size': 2000}
    columns = ['id_left', 'id_right', 'label']
    return DataPack(relation=relation, content=content, context=ctx, columns=columns)

def test_length(data_pack):
    num_examples = 2
    assert len(data_pack) == num_examples

def test_content(data_pack):
    assert data_pack.content['qid0'] == [1, 2]

def test_save_load(data_pack):
    dirpath = '.tmpdir'
    data_pack.save(dirpath)
    dp = load_datapack(dirpath)
    with pytest.raises(FileExistsError):
        data_pack.save(dirpath)
    assert len(data_pack) == 2
    shutil.rmtree(dirpath)
