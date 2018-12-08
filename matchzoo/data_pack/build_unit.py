"""Build a :class:`processor_units.VocabularyUnit` given `data_pack`."""
from tqdm import tqdm

from . import DataPack
from matchzoo.processor_units import StatefulProcessorUnit


def build_unit_from_datapack(unit: StatefulProcessorUnit, data_pack: DataPack,
                             flatten: bool = True, verbose: int = 1
                             ) -> StatefulProcessorUnit:
    """
    Build a :class:`StatefulProcessorUnit` from a :class:`DataPack` object.

    :param unit: :class:`StatefulProcessorUnit` object to be built.
    :param data_pack: The input :class:`DataPack` object.
    :param flatten: To flatten the datapack or not.
    :param verbose: Verbosity.
    :return: A built :class:`StatefulProcessorUnit` object.

    """
    corpus = []
    if flatten:
        data_pack.apply_on_text(corpus.extend, verbose=verbose)
    else:
        data_pack.apply_on_text(corpus.append, verbose=verbose)
    if verbose:
        description = 'Build ' + unit.__class__.__name__ + ' from a datapack.'
        corpus = tqdm(corpus, desc=description)
    unit.fit(corpus)
    return unit
