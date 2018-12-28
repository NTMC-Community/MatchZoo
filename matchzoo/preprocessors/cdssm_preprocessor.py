"""CDSSM Preprocessor."""

import logging

from tqdm import tqdm

from matchzoo import build_vocab_unit
from matchzoo import chain_transform
from matchzoo import DataPack
from matchzoo import engine
from matchzoo import processor_units

logger = logging.getLogger(__name__)
tqdm.pandas()


class CDSSMPreprocessor(engine.BasePreprocessor):
    """CDSSM Model preprocessor."""

    def __init__(self, text_len: int = 10, with_word_hashing=True):
        """
        CDSSM Model preprocessor.

        The word hashing step could eats up a lot of memory. To workaround
        this problem, set `with_word_hashing` to `False` and use a
        :class:`matchzoo.DynamicDataGenerator` with a
        :class:`matchzoo.processor_units.WordHashingUnit`.

        :param text_len: Fixed text length, cut original text if it is longer
         or pad if shorter.
        :param with_word_hashing: Include a word hashing step if `True`.

        Example:
            >>> import matchzoo as mz
            >>> train_data = mz.datasets.toy.load_data()
            >>> test_data = mz.datasets.toy.load_data(stage='test')
            >>> cdssm_preprocessor = mz.preprocessors.CDSSMPreprocessor()
            >>> train_data_processed = cdssm_preprocessor.fit_transform(
            ...     train_data
            ... )
            >>> type(train_data_processed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>
            >>> test_data_transformed = cdssm_preprocessor.transform(test_data)
            >>> type(test_data_transformed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>

        """
        super().__init__()
        self._text_len = text_len
        self._with_word_hashing = with_word_hashing

    def fit(self, data_pack: DataPack, verbose=1):
        """
        Fit pre-processing context for transformation.

        :param verbose: Verbosity.
        :param data_pack: Data_pack to be preprocessed.
        :return: class:`CDSSMPreprocessor` instance.
        """
        units = self._default_processor_units()
        units.append(processor_units.NgramLetterUnit())
        data_pack = data_pack.apply_on_text(chain_transform(units),
                                            verbose=verbose)
        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)

        self._context['vocab_unit'] = vocab_unit
        num_letter = len(vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = (self._text_len, num_letter)
        return self

    @engine.validate_context
    def transform(self, data_pack: DataPack, verbose=1) -> DataPack:
        """
        Apply transformation on data, create `letter-ngram` representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        units = self._default_processor_units()
        units.append(processor_units.FixedLengthUnit(
            text_length=self._text_len, pad_value='0'))
        units.append(processor_units.NgramLetterUnit(reduce_dim=False))
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
        ]
