"""DSSM Preprocessor."""

import logging

from tqdm import tqdm

from matchzoo import engine, processor_units
from matchzoo import DataPack
from matchzoo import chain_transform, build_vocab_unit

logger = logging.getLogger(__name__)
tqdm.pandas()


class DSSMPreprocessor(engine.BasePreprocessor):
    """DSSM Model preprocessor."""

    def __init__(self, with_word_hashing: bool = True):
        """
        DSSM Model preprocessor.

        The word hashing step could eats up a lot of memory. To workaround
        this problem, set `with_word_hashing` to `False` and use  a
        :class:`matchzoo.DynamicDataGenerator` with a
        :class:`matchzoo.processor_units.WordHashingUnit`.

        :param with_word_hashing: Include a word hashing step if `True`.

        Example:
            >>> import matchzoo as mz
            >>> train_data = mz.datasets.toy.load_data()
            >>> test_data = mz.datasets.toy.load_data(stage='test')
            >>> dssm_preprocessor = mz.preprocessors.DSSMPreprocessor()
            >>> train_data_processed = dssm_preprocessor.fit_transform(
            ...     train_data
            ... )
            >>> type(train_data_processed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>
            >>> test_data_transformed = dssm_preprocessor.transform(test_data)
            >>> type(test_data_transformed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>

        """
        super().__init__()
        self._with_word_hashing = with_word_hashing

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        Fit pre-processing context for transformation.

        :param verbose: Verbosity.
        :param data_pack: data_pack to be preprocessed.
        :return: class:`DSSMPreprocessor` instance.
        """
        units = self._default_processor_units()
        data_pack = data_pack.apply_on_text(chain_transform(units),
                                            verbose=verbose)
        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)

        self._context['vocab_unit'] = vocab_unit
        triletter_dim = len(vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = [(triletter_dim,), (triletter_dim,)]
        return self

    @engine.validate_context
    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create `tri-letter` representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        units = self._default_processor_units()
        if self._with_word_hashing:
            term_index = self._context['vocab_unit'].state['term_index']
            units.append(processor_units.WordHashingUnit(term_index))
        data_pack.apply_on_text(chain_transform(units), inplace=True,
                                verbose=verbose)
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
