"""Build a :class:`processor_units.VocabularyUnit` given `data_pack`."""

from tqdm import tqdm

from . import DataPack
from matchzoo import processor_units


def build_vocab(data_pack: DataPack) -> processor_units.VocabularyUnit:
    """
    Build a :class:`processor_units.VocabularyUnit` given `data_pack`.

    The `data_pack` should be preprocessed forehand, and each item in
    `text_left` and `text_right` columns of the `data_pack` should be a list
    of tokens.

    :param data_pack: The :class:`DataPack` to build vocabulary upon.
    :return: A built vocabulary unit.
    """
    vocab = []
    data_pack.apply_on_text(vocab.extend)
    vocab_unit = processor_units.VocabularyUnit()
    vocab_unit.fit(tqdm(vocab, desc='Fitting vocabulary unit.'))
    return vocab_unit
