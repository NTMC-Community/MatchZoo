import pytest
import numpy as np

import matchzoo as mz


@pytest.fixture
def term_index_input():
    return {'G': 1, 'C': 2, 'D': 3, 'A': 4, '[PAD]': 0}


def test_embedding(term_index_input):
    embed = mz.datasets.toy.load_embedding()
    embed.build(term_index_input)
    assert embed.embedding_dim == 10
    assert embed.index_state == {4: 1, 2: 1, 3: 1, 1: 2, 0: 0}
    assert (embed.embedding_mat[0] == np.array([0] * 10)).all()
    assert (np.abs(embed.embedding_mat[2] - \
                   (np.array(range(10)) * 0.1 + 0.1)) < 1e-6).all()

    embed = mz.datasets.toy.load_embedding()
    embed.build(term_index_input)
    assert embed.embedding_dim == 10
    assert embed.index_state == {4: 1, 2: 1, 3: 1, 1: 2, 0: 0}
    assert (embed.embedding_mat[0] == np.array([0] * 10)).all()
    assert (np.abs(embed.embedding_mat[2] - \
                   (np.array(range(10)) * 0.1 + 0.1)) < 1e-6).all()

    embed = mz.datasets.toy.load_embedding()
    del term_index_input['[PAD]']
    embed.build(term_index_input)
    assert embed.embedding_dim == 10
    assert embed.index_state == {4: 1, 2: 1, 3: 1, 1: 2, 0: 0}
    assert (embed.embedding_mat[0] == np.array([0] * 10)).all()
    assert (np.abs(embed.embedding_mat[2] - \
                   (np.array(range(10)) * 0.1 + 0.1)) < 1e-6).all()
