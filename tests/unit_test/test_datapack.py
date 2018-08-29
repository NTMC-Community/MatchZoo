from matchzoo.datapack import DataPack, load_datapack

import pytest
import shutil

import pandas as pd

@pytest.fixture
def data_pack():
    relation = [['qid0', 'did0', 1], ['qid1', 'did1', 0]]
    left_data = [['qid0', [1, 2]], ['qid1', [2, 3]]]
    right_data = [['did0', [2, 3, 4]], ['did1', [3, 4, 5]]]
    ctx = {'vocab_size': 2000}
    relation_columns = ['id_left', 'id_right', 'label']
    left_columns = ['id_left', 'text_left']
    right_columns = ['id_right', 'text_right']
    relation_df = pd.DataFrame(relation, columns=relation_columns)
    left_df = pd.DataFrame(left_data, columns=left_columns)
    left_df.set_index('id_left', inplace=True)
    right_df = pd.DataFrame(right_data, columns=right_columns)
    right_df.set_index('id_right', inplace=True)
    return DataPack(relation=relation_df,
                    left_data=left_df,
                    right_data=right_df,
                    context=ctx
    )

def test_length(data_pack):
    num_examples = 2
    assert len(data_pack) == num_examples

def test_getter(data_pack):
    assert data_pack.relation.iloc[0].values.tolist() == ['qid0', 'did0', 1]
    assert data_pack.relation.iloc[1].values.tolist() == ['qid1', 'did1', 0]
    assert data_pack.left_data.loc['qid0', 'text_left'] == [1, 2]
    assert data_pack.right_data.loc['did1', 'text_right'] == [3, 4, 5]

def test_setter(data_pack):
    data = [['id0', [1]], ['id1', [2]]]
    left_data = pd.DataFrame(data, columns=['id_left', 'text_left'])
    left_data.set_index('id_left', inplace=True)
    data_pack.left_data = left_data
    assert data_pack.left_data.loc['id0', 'text_left'] == [1]
    right_data = pd.DataFrame(data, columns=['id_right', 'text_right'])
    right_data.set_index('id_right', inplace=True)
    data_pack.right_data = right_data
    assert data_pack.right_data.loc['id0', 'text_right'] == [1]

def test_save_load(data_pack):
    dirpath = '.tmpdir'
    data_pack.save(dirpath)
    dp = load_datapack(dirpath)
    with pytest.raises(FileExistsError):
        data_pack.save(dirpath)
    assert len(data_pack) == 2
    assert len(dp) == 2
    shutil.rmtree(dirpath)

d = data_pack()
print(d.left_data)
d.left_data.at['qid0', 'text_left'] = [1000]
print(d.left_data)
