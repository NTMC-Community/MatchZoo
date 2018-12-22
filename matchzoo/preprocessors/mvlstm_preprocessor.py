"""MVLSTM Preprocessor."""

import logging

from tqdm import tqdm

from matchzoo import engine, processor_units
from matchzoo import DataPack
from matchzoo import chain_transform, build_vocab_unit

logger = logging.getLogger(__name__)
tqdm.pandas()


class MVLSTMPreprocessor(engine.BasePreprocessor):
    """MVLSTMModel preprocessor."""

    def __init__(self, fixed_length: list = [10, 10]):
        """
        MVLSTM Model preprocessor.

        :param fixed_length: The fixed length of 'text_left' 
            and 'text_right'.

        Example:
            >>> import matchzoo as mz
            >>> train_data = mz.datasets.toy.load_train_classify_data()
            >>> test_data = mz.datasets.toy.load_test_classify_data()
            >>> mvlstm_preprocessor = mz.preprocessors.MVLSTMPreprocessor()
            >>> train_data_processed = mvlstm_preprocessor.fit_transform(train_data)
            >>> type(train_data_processed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>
            >>> test_data_transformed = mvlstm_preprocessor.transform(test_data)
            >>> type(test_data_transformed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>

        """
        super().__init__()
        self._fixed_length = fixed_length
        self._left_fixedlength_unit = processor_units.FixedLengthUnit(
            self._fixed_length[0], pad_mode='post')
        self._right_fixedlength_unit = processor_units.FixedLengthUnit(
            self._fixed_length[1], pad_mode='post')
        self._VocabularyUnit = processor_units.VocabularyUnit

    def fit(self, data_pack: DataPack, verbose=1):
        """
        Fit pre-processing context for transformation.

        :param verbose: Verbosity.
        :param data_pack: data_pack to be preprocessed.
        :return: class:`MVLSTMPreprocessor` instance.
        """
        units = self._default_processor_units()
        data_pack = data_pack.apply_on_text(chain_transform(units),
                                            verbose=verbose)
        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)

        self._context['embedding_input_dim'] = len(
                    vocab_unit.state['term_index']) + 1 
        self._context['vocab_unit'] = vocab_unit
        self._context['input_shapes'] = [(self._fixed_length[0],),
                                         (self._fixed_length[1],)]
        return self

    @engine.validate_context
    def transform(self, data_pack: DataPack, verbose=1) -> DataPack:
        """ 
        Apply transformation on data, create fixed length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        units = self._default_processor_units()
        data_pack.apply_on_text(chain_transform(units), inplace=True,
                                verbose=verbose)

        data_pack.append_text_length(inplace=True)
        data_pack.apply_on_text(self._left_fixedlength_unit.transform,
                                mode='both', inplace=True, verbose=verbose)

        data_pack.apply_on_text(self._context['vocab_unit'].transform,
                                mode='both', inplace=True, verbose=verbose)
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
