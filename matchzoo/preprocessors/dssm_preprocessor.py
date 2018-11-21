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
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_train_classify_data()
        >>> test_data = mz.datasets.toy.load_test_classify_data()
        >>> dssm_preprocessor = mz.preprocessors.DSSMPreprocessor()
        >>> train_data_processed = dssm_preprocessor.fit_transform(train_data)
        >>> type(train_data_processed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        >>> test_data_transformed = dssm_preprocessor.transform(test_data)
        >>> type(test_data_transformed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>

    """

    def __init__(self, with_word_hashing=True):
        super().__init__()
        self._with_word_hashing = with_word_hashing

    def fit(self, data_pack: DataPack):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :return: class:`DSSMPreprocessor` instance.
        """
        units = self._default_processor_units()
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
        units = self._default_processor_units()
        if self._with_word_hashing:
            term_index = self._context['term_index']
            units.append(processor_units.WordHashingUnit(term_index))
        data_pack.apply_on_text(chain_transform(units), inplace=True)
        return data_pack

    @classmethod
    def _default_processor_units(cls) -> list:
        """Prepare needed process units."""
        return [
            processor_units.TokenizeUnit(),
            processor_units.LowercaseUnit(),
            processor_units.PuncRemovalUnit(),
            processor_units.StopRemovalUnit(),
            processor_units.NgramLetterUnit(),
        ]
