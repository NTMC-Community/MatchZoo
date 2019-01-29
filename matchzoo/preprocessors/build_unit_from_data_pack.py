"""Build unit from data pack."""

from tqdm import tqdm

import matchzoo as mz
from .units import StatefulUnit


def build_unit_from_data_pack(
    unit: StatefulUnit,
    data_pack: mz.DataPack, mode: str = 'both',
    flatten: bool = True, verbose: int = 1
) -> StatefulUnit:
    """
    Build a :class:`StatefulUnit` from a :class:`DataPack` object.

    :param unit: :class:`StatefulUnit` object to be built.
    :param data_pack: The input :class:`DataPack` object.
    :param mode: One of 'left', 'right', and 'both', to determine the source
            data for building the :class:`VocabularyUnit`.
    :param flatten: Flatten the datapack or not. `True` to organize the
        :class:`DataPack` text as a list, and `False` to organize
        :class:`DataPack` text as a list of list.
    :param verbose: Verbosity.
    :return: A built :class:`StatefulUnit` object.

    """
    corpus = []
    if flatten:
        data_pack.apply_on_text(corpus.extend, mode=mode, verbose=verbose)
    else:
        data_pack.apply_on_text(corpus.append, mode=mode, verbose=verbose)
    if verbose:
        description = 'Building ' + unit.__class__.__name__ + \
                      ' from a datapack.'
        corpus = tqdm(corpus, desc=description)
    unit.fit(corpus)
    return unit
