"""DSSM Preprocessor."""

import logging

from tqdm import tqdm

from matchzoo import engine, preprocessors, processor_units
from matchzoo import DataPack
from matchzoo import chain_transform, build_vocab

logger = logging.getLogger(__name__)
tqdm.pandas()


class DSSMPreprocessor(engine.BasePreprocessor):
    """
    DSSM preprocessor helper.

    TODO: NEED REFACTORING.

    Example:
        >>> train_inputs = [
        ...     ("id0", "id1", "beijing", "Beijing is capital of China", 1),
        ...     ("id0", "id2", "beijing", "China is in east Asia", 0),
        ...     ("id0", "id3", "beijing", "Summer in Beijing is hot.", 1)
        ... ]
        >>> dssm_preprocessor = DSSMPreprocessor()
        >>> rv_train = dssm_preprocessor.fit_transform(
        ...     train_inputs,
        ...     stage='train')
        >>> type(rv_train)
        <class 'matchzoo.data_pack.DataPack'>
        >>> test_inputs = [("id0",
        ...                 "id4",
        ...                 "beijing",
        ...                 "I visted beijing yesterday.")]
        >>> rv_test = dssm_preprocessor.fit_transform(
        ...     test_inputs,
        ...     stage='predict')
        >>> type(rv_test)
        <class 'matchzoo.data_pack.DataPack'>

    """

    def fit(self, data_pack: DataPack):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :return: class:`DSSMPreprocessor` instance.
        """
        units = self._preprocess_units()
        data_pack = data_pack.apply_on_text(chain_transform(units))
        vocab_unit = build_vocab(data_pack)

        self._context.update(vocab_unit.state)
        triletter_dim = len(vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = [(triletter_dim,), (triletter_dim,)]

        return self

    @engine.validate_context
    def transform(self, data_pack: DataPack) -> DataPack:
        """
        Apply transformation on data, create `tri-letter` representation.

        :param data_pack: Inputs to be preprocessed.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        term_index = self._context['term_index']
        hashing_unit = processor_units.WordHashingUnit(term_index)
        units = self._preprocess_units() + [hashing_unit]
        data_pack.apply_on_text(chain_transform(units), inplace=True)
        return data_pack

    @classmethod
    def _preprocess_units(cls) -> list:
        """Prepare needed process units."""
        return [
            processor_units.TokenizeUnit(),
            processor_units.LowercaseUnit(),
            processor_units.PuncRemovalUnit(),
            processor_units.StopRemovalUnit(),
            processor_units.NgramLetterUnit(),
        ]
