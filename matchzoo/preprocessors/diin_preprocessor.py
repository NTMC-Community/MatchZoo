"""DIIN Preprocessor."""

from tqdm import tqdm
import pandas as pd

from matchzoo.engine.base_preprocessor import BasePreprocessor
from matchzoo import DataPack
from .build_vocab_unit import build_vocab_unit
from .chain_transform import chain_transform
from . import units

tqdm.pandas()


class DIINPreprocessor(BasePreprocessor):
    """DIIN Model preprocessor."""

    def __init__(self,
                 fixed_length_left: int = 10,
                 fixed_length_right: int = 10,
                 fixed_length_word: int = 5):
        """
        DIIN Model preprocessor.

        :param fixed_length_left: Integer, maximize length of :attr:'left' in
            the data_pack.
        :param fixed_length_right: Integer, maximize length of :attr:'right' in
            the data_pack.
        :param fixed_length_word: Integer, maximize length of each word.

        Example:
            >>> import matchzoo as mz
            >>> train_data = mz.datasets.toy.load_data()
            >>> test_data = mz.datasets.toy.load_data(stage='test')
            >>> diin_preprocessor = mz.preprocessors.DIINPreprocessor(
            ...     fixed_length_left=5,
            ...     fixed_length_right=5,
            ...     fixed_length_word=3,
            ... )
            >>> diin_preprocessor = diin_preprocessor.fit(
            ...     train_data, verbose=0)
            >>> diin_preprocessor.context['input_shapes']
            [(5,), (5,), (5, 3), (5, 3), (5,), (5,)]
            >>> diin_preprocessor.context['vocab_size']
            893
            >>> train_data_processed = diin_preprocessor.transform(
            ...     train_data, verbose=0)
            >>> type(train_data_processed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>
            >>> test_data_processed = diin_preprocessor.transform(
            ...     test_data, verbose=0)
            >>> type(test_data_processed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>

        """
        super().__init__()
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._fixed_length_word = fixed_length_word
        self._left_fixedlength_unit = units.FixedLength(
            self._fixed_length_left,
            pad_value='0',
            pad_mode='post'
        )
        self._right_fixedlength_unit = units.FixedLength(
            self._fixed_length_right,
            pad_value='0',
            pad_mode='post'
        )
        self._units = self._default_units()

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :param verbose: Verbosity.
        :return: class:'DIINPreprocessor' instance.
        """
        func = chain_transform(self._units)
        data_pack = data_pack.apply_on_text(func, mode='both', verbose=verbose)

        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)
        vocab_size = len(vocab_unit.state['term_index'])
        self._context['vocab_unit'] = vocab_unit
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size

        data_pack = data_pack.apply_on_text(
            units.NgramLetter(ngram=1, reduce_dim=True).transform,
            mode='both', verbose=verbose)
        char_unit = build_vocab_unit(data_pack, verbose=verbose)
        self._context['char_unit'] = char_unit

        self._context['input_shapes'] = [
            (self._fixed_length_left,),
            (self._fixed_length_right,),
            (self._fixed_length_left, self._fixed_length_word,),
            (self._fixed_length_right, self._fixed_length_word,),
            (self._fixed_length_left,),
            (self._fixed_length_right,)
        ]
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:'DataPack' object.
        """
        data_pack = data_pack.copy()
        data_pack.apply_on_text(
            chain_transform(self._units),
            mode='both', inplace=True, verbose=verbose)

        # Process character representation
        data_pack.apply_on_text(
            units.NgramLetter(ngram=1, reduce_dim=False).transform,
            rename=('char_left', 'char_right'),
            mode='both', inplace=True, verbose=verbose)
        char_index_dict = self._context['char_unit'].state['term_index']
        left_charindex_unit = units.CharacterIndex(
            char_index_dict, self._fixed_length_left, self._fixed_length_word)
        right_charindex_unit = units.CharacterIndex(
            char_index_dict, self._fixed_length_right, self._fixed_length_word)
        data_pack.left['char_left'] = data_pack.left['char_left'].apply(
            left_charindex_unit.transform)
        data_pack.right['char_right'] = data_pack.right['char_right'].apply(
            right_charindex_unit.transform)

        # Process word representation
        data_pack.apply_on_text(
            self._context['vocab_unit'].transform,
            mode='both', inplace=True, verbose=verbose)

        # Process exact match representation
        frame = data_pack.relation.join(
            data_pack.left, on='id_left', how='left'
        ).join(data_pack.right, on='id_right', how='left')
        left_exactmatch_unit = units.WordExactMatch(
            self._fixed_length_left, match='text_left', to_match='text_right')
        right_exactmatch_unit = units.WordExactMatch(
            self._fixed_length_right, match='text_right', to_match='text_left')
        data_pack.relation['match_left'] = frame.apply(
            left_exactmatch_unit.transform, axis=1)
        data_pack.relation['match_right'] = frame.apply(
            right_exactmatch_unit.transform, axis=1)

        data_pack.apply_on_text(
            self._left_fixedlength_unit.transform,
            mode='left', inplace=True, verbose=verbose)
        data_pack.apply_on_text(
            self._right_fixedlength_unit.transform,
            mode='right', inplace=True, verbose=verbose)

        return data_pack
