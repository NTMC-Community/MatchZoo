"""Embedding data loader."""

from pathlib import Path

import keras

import matchzoo as mz

_glove_embedding_url = "http://nlp.stanford.edu/data/glove.6B.zip"


def load_glove_embedding(dimension: int = 50) -> mz.embedding.Embedding:
    """
    Return the pretrained glove embedding.

    :param dimension: the size of embedding dimension, the value can only be
        50, 100, or 300.
    :return: The :class:`mz.embedding.Embedding` object.
    """
    ref_path = keras.utils.data_utils.get_file('glove_embedding',
                                               _glove_embedding_url,
                                               extract=True)
    file_name = 'glove.6B.' + str(dimension) + 'd.txt'
    file_path = Path(ref_path).parent.joinpath(file_name)
    return mz.embedding.load_from_file(file_path=file_path, mode='glove')
