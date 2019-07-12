"""Bert Preprocessor."""

from tqdm import tqdm

from . import units
from .chain_transform import chain_transform
from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from .build_vocab_unit import built_bert_vocab_unit
from .build_unit_from_data_pack import build_unit_from_data_pack

tqdm.pandas()


class BertPreprocessor(BasePreprocessor):
    """Bert-base Model preprocessor."""

    def __init__(self, bert_vocab_path: str,
                 fixed_length_left: int = 30,
                 fixed_length_right: int = 30,
                 filter_mode: str = 'df',
                 filter_low_freq: float = 2,
                 filter_high_freq: float = float('inf'),
                 remove_stop_words: bool = False,
                 lower_case: bool = True,
                 chinese_version: bool = False,
                 ):
        """
        Bert-base Model preprocessor.

        Example:
            >>> import matchzoo as mz
            >>> train_data = mz.datasets.toy.load_data()
            >>> test_data = mz.datasets.toy.load_data(stage='test')
            >>> # The argument 'bert_vocab_path' must feed the bert vocab path
            >>> bert_preprocessor = mz.preprocessors.BertPreprocessor(
            ...     bert_vocab_path=
            ...     'matchzoo/datasets/bert_resources/uncased_vocab_100.txt')
            >>> train_data_processed = bert_preprocessor.fit_transform(
            ...     train_data)
            >>> test_data_processed = bert_preprocessor.transform(test_data)

        """
        super().__init__()
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._bert_vocab_path = bert_vocab_path
        self._left_fixedlength_unit = units.FixedLength(
            self._fixed_length_left,
            pad_mode='post'
        )
        self._right_fixedlength_unit = units.FixedLength(
            self._fixed_length_right,
            pad_mode='post'
        )
        self._filter_unit = units.FrequencyFilter(
            low=filter_low_freq,
            high=filter_high_freq,
            mode=filter_mode
        )
        self._units = self._default_units()
        self._vocab_unit = built_bert_vocab_unit(self._bert_vocab_path)

        if chinese_version:
            self._units.insert(1, units.ChineseTokenize())
        if lower_case:
            self._units.append(units.Lowercase())
            self._units.append(units.StripAccent())
        self._units.append(units.WordPieceTokenize(
            self._vocab_unit.state['term_index']))
        if remove_stop_words:
            self._units.append(units.StopRemoval())

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        Fit pre-processing context for transformation.

        :param verbose: Verbosity.
        :param data_pack: Data_pack to be preprocessed.
        :return: class:`BertPreprocessor` instance.
        """
        data_pack = data_pack.apply_on_text(chain_transform(self._units),
                                            verbose=verbose)
        fitted_filter_unit = build_unit_from_data_pack(self._filter_unit,
                                                       data_pack,
                                                       flatten=False,
                                                       mode='right',
                                                       verbose=verbose)
        self._context['filter_unit'] = fitted_filter_unit
        self._context['vocab_unit'] = self._vocab_unit
        vocab_size = len(self._vocab_unit.state['term_index'])
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size
        self._context['input_shapes'] = [(self._fixed_length_left,),
                                         (self._fixed_length_right,)]
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create fixed length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        data_pack.apply_on_text(chain_transform(self._units), inplace=True,
                                verbose=verbose)

        data_pack.apply_on_text(self._context['filter_unit'].transform,
                                mode='right', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._context['vocab_unit'].transform,
                                mode='both', inplace=True, verbose=verbose)
        data_pack.append_text_length(inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._left_fixedlength_unit.transform,
                                mode='left', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._right_fixedlength_unit.transform,
                                mode='right', inplace=True, verbose=verbose)

        max_len_left = self._fixed_length_left
        max_len_right = self._fixed_length_right

        data_pack.left['length_left'] = \
            data_pack.left['length_left'].apply(
                lambda val: min(val, max_len_left))

        data_pack.right['length_right'] = \
            data_pack.right['length_right'].apply(
                lambda val: min(val, max_len_right))
        return data_pack

    @classmethod
    def _default_units(cls) -> list:
        """Prepare needed process units."""
        return [
            units.BertClean(),
            units.BasicTokenize()
        ]
