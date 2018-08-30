from matchzoo.datapack import DataPack, load_datapack

import pytest
import shutil

import pandas as pd

@pytest.fixture
def data_pack():
    relation_data = [['qid0', 'did0', 1], ['qid1', 'did1', 0]]
    left_data = [['qid0', [1, 2]], ['qid1', [2, 3]]]
    right_data = [['did0', [2, 3, 4]], ['did1', [3, 4, 5]]]
    ctx = {'vocab_size': 2000}
    relation_columns = ['id_left', 'id_right', 'label']
    left_columns = ['id_left', 'text_left']
    right_columns = ['id_right', 'text_right']
    relation = pd.DataFrame(relation_data, columns=relation_columns)
    left = pd.DataFrame(left_data, columns=left_columns)
    left.set_index('id_left', inplace=True)
    right = pd.DataFrame(right_data, columns=right_columns)
    right.set_index('id_right', inplace=True)
    return DataPack(relation=relation,
                    left=left,
                    right=right,
                    context=ctx
    )

def test_length(data_pack):
    num_examples = 2
    assert len(data_pack) == num_examples

def test_getter(data_pack):
    assert data_pack.relation.iloc[0].values.tolist() == ['qid0', 'did0', 1]
    assert data_pack.relation.iloc[1].values.tolist() == ['qid1', 'did1', 0]
    assert data_pack.left.loc['qid0', 'text_left'] == [1, 2]
    assert data_pack.right.loc['did1', 'text_right'] == [3, 4, 5]

def test_setter(data_pack):
    data = [['id0', [1]], ['id1', [2]]]
    left = pd.DataFrame(data, columns=['id_left', 'text_left'])
    left.set_index('id_left', inplace=True)
    data_pack.left = left
    assert data_pack.left.loc['id0', 'text_left'] == [1]
    right = pd.DataFrame(data, columns=['id_right', 'text_right'])
    right.set_index('id_right', inplace=True)
    data_pack.right = right
    assert data_pack.right.loc['id0', 'text_right'] == [1]

def test_save_load(data_pack):
    dirpath = '.tmpdir'
    data_pack.save(dirpath)
    dp = load_datapack(dirpath)
    with pytest.raises(FileExistsError):
        data_pack.save(dirpath)
    assert len(data_pack) == 2
    assert len(dp) == 2
    shutil.rmtree(dirpath)
