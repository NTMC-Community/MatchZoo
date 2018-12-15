import shutil

import pandas as pd
import pytest

from matchzoo import DataPack, load_data_pack


@pytest.fixture
def data_pack():
    relation = [['qid0', 'did0', 1], ['qid1', 'did1', 0]]
    left = [['qid0', [1, 2]], ['qid1', [2, 3]]]
    right = [['did0', [2, 3, 4]], ['did1', [3, 4, 5]]]
    relation = pd.DataFrame(relation, columns=['id_left', 'id_right', 'label'])
    left = pd.DataFrame(left, columns=['id_left', 'text_left'])
    left.set_index('id_left', inplace=True)
    right = pd.DataFrame(right, columns=['id_right', 'text_right'])
    right.set_index('id_right', inplace=True)
    return DataPack(relation=relation,
                    left=left,
                    right=right)


def test_length(data_pack):
    num_examples = 2
    assert len(data_pack) == num_examples


def test_getter(data_pack):
    assert data_pack.relation.iloc[0].values.tolist() == ['qid0', 'did0', 1]
    assert data_pack.relation.iloc[1].values.tolist() == ['qid1', 'did1', 0]
    assert data_pack.left.loc['qid0', 'text_left'] == [1, 2]
    assert data_pack.right.loc['did1', 'text_right'] == [3, 4, 5]


def test_save_load(data_pack):
    dirpath = '.tmpdir'
    data_pack.save(dirpath)
    dp = load_data_pack(dirpath)
    with pytest.raises(FileExistsError):
        data_pack.save(dirpath)
    assert len(data_pack) == 2
    assert len(dp) == 2
    shutil.rmtree(dirpath)
