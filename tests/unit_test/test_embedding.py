import pytest

import matchzoo as mz


@pytest.fixture
def term_index():
    return {'G': 1, 'C': 2, 'D': 3, 'A': 4, '[PAD]': 0}


def test_embedding(term_index):
    embed = mz.embedding.load_from_file(mz.datasets.embeddings.EMBED_RANK)
    matrix = embed.build_matrix(term_index)
    assert matrix.shape == (len(term_index) + 1, 50)
    embed = mz.embedding.load_from_file(mz.datasets.embeddings.EMBED_10_GLOVE,
                                        mode='glove')
    matrix = embed.build_matrix(term_index)
    assert matrix.shape == (len(term_index) + 1, 10)
    assert embed.input_dim == 5
