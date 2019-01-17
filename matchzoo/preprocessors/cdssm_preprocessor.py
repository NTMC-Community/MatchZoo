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

    def __init__(self,
                 fixed_length_left: int = 10,
                 fixed_length_right: int = 40,
                 with_word_hashing: bool = True):
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
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._left_fixedlength_unit = processor_units.FixedLengthUnit(
            self._fixed_length_left,
            pad_value='0', pad_mode='post'
        )
        self._right_fixedlength_unit = processor_units.FixedLengthUnit(
            self._fixed_length_right,
            pad_value='0', pad_mode='post'
        )
        self._with_word_hashing = with_word_hashing

    def fit(self, data_pack: DataPack, verbose: int = 1):
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
        vocab_size = len(vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = [
            (self._fixed_length_left, vocab_size),
            (self._fixed_length_right, vocab_size)
        ]
        return self

    @engine.validate_context
    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create `letter-ngram` representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        units = self._default_processor_units()
        data_pack.apply_on_text(chain_transform(units), inplace=True,
                                verbose=verbose)
        data_pack.apply_on_text(self._left_fixedlength_unit.transform,
                                mode='left', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._right_fixedlength_unit.transform,
                                mode='right', inplace=True, verbose=verbose)
        post_units = [processor_units.NgramLetterUnit(reduce_dim=False)]
        if self._with_word_hashing:
            term_index = self._context['vocab_unit'].state['term_index']
            post_units.append(processor_units.WordHashingUnit(term_index))
        data_pack.apply_on_text(chain_transform(post_units),
                                inplace=True, verbose=verbose)
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
