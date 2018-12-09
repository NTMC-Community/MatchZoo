"""Build a :class:`processor_units.VocabularyUnit` given `data_pack`."""
from tqdm import tqdm

from . import DataPack
from matchzoo import processor_units


def build_vocab_unit(data_pack: DataPack,
                     verbose=1) -> processor_units.VocabularyUnit:
    """
    Build a :class:`processor_units.VocabularyUnit` given `data_pack`.

    The `data_pack` should be preprocessed forehand, and each item in
    `text_left` and `text_right` columns of the `data_pack` should be a list
    of tokens.

    :param data_pack: The :class:`DataPack` to build vocabulary upon.
    :param verbose: Verbosity.
    :return: A built vocabulary unit.

    """
    vocab_unit = processor_units.VocabularyUnit()
    vocab_unit = build_unit_from_data_pack(vocab_unit, data_pack, True,
                                           verbose)
    return vocab_unit


def build_unit_from_data_pack(unit: processor_units.StatefulProcessorUnit,
                              data_pack: DataPack, flatten: bool = True,
                              verbose: int = 1
                              ) -> processor_units.StatefulProcessorUnit:
    """
    Build a :class:`StatefulProcessorUnit` from a :class:`DataPack` object.

    :param unit: :class:`StatefulProcessorUnit` object to be built.
    :param data_pack: The input :class:`DataPack` object.
    :param flatten: Flatten the datapack or not. `True` to organize the
        :class:`DataPack` text as a list, and `False` to organize
        :class:`DataPack` text as a list of list.
    :param verbose: Verbosity.
    :return: A built :class:`StatefulProcessorUnit` object.

    """
    corpus = []
    if flatten:
        data_pack.apply_on_text(corpus.extend, verbose=verbose)
    else:
        data_pack.apply_on_text(corpus.append, verbose=verbose)
    if verbose:
        description = 'Building ' + unit.__class__.__name__ + \
                      ' from a datapack.'
        corpus = tqdm(corpus, desc=description)
    unit.fit(corpus)
    return unit
