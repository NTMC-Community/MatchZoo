"""Build a :class:`processor_units.VocabularyUnit` given `data_pack`."""

from tqdm import tqdm

from . import DataPack
from matchzoo import processor_units


def build_vocab(
    data_pack: DataPack, verbose=1
) -> processor_units.VocabularyUnit:
    """
    Build a :class:`processor_units.VocabularyUnit` given `data_pack`.

    The `data_pack` should be preprocessed forehand, and each item in
    `text_left` and `text_right` columns of the `data_pack` should be a list
    of tokens.

    :param data_pack: The :class:`DataPack` to build vocabulary upon.
    :param verbose: Verbosity.

    :return: A built vocabulary unit.
    """
    vocab = []
    data_pack.apply_on_text(vocab.extend, verbose=verbose)
    vocab_unit = processor_units.VocabularyUnit()
    if verbose:
        vocab = tqdm(vocab, desc='Fitting vocabulary unit.')
    vocab_unit.fit(vocab)
    return vocab_unit
