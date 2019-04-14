"""CDSSM Preprocessor."""

from tqdm import tqdm

from . import units
from .chain_transform import chain_transform
from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from .build_vocab_unit import build_vocab_unit

tqdm.pandas()


class CDSSMPreprocessor(BasePreprocessor):
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
        :class:`matchzoo.preprocessor.units.WordHashing`.

        TODO: doc here.

        :param with_word_hashing: Include a word hashing step if `True`.

        Example:
            >>> import matchzoo as mz
            >>> train_data = mz.datasets.toy.load_data()
            >>> test_data = mz.datasets.toy.load_data(stage='test')
            >>> cdssm_preprocessor = mz.preprocessors.CDSSMPreprocessor()
            >>> train_data_processed = cdssm_preprocessor.fit_transform(
            ...     train_data, verbose=0
            ... )
            >>> type(train_data_processed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>
            >>> test_data_transformed = cdssm_preprocessor.transform(test_data,
            ...                                                      verbose=0)
            >>> type(test_data_transformed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>

        """
        super().__init__()
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._left_fixedlength_unit = units.FixedLength(
            self._fixed_length_left,
            pad_value='0', pad_mode='post'
        )
        self._right_fixedlength_unit = units.FixedLength(
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
        fit_units = self._default_units() + [units.NgramLetter()]
        func = chain_transform(fit_units)
        data_pack = data_pack.apply_on_text(func, verbose=verbose)
        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)

        self._context['vocab_unit'] = vocab_unit
        vocab_size = len(vocab_unit.state['term_index'])
        self._context['input_shapes'] = [
            (self._fixed_length_left, vocab_size),
            (self._fixed_length_right, vocab_size)
        ]
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create `letter-ngram` representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        func = chain_transform(self._default_units())
        data_pack.apply_on_text(func, inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._left_fixedlength_unit.transform,
                                mode='left', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._right_fixedlength_unit.transform,
                                mode='right', inplace=True, verbose=verbose)
        post_units = [units.NgramLetter(reduce_dim=False)]
        if self._with_word_hashing:
            term_index = self._context['vocab_unit'].state['term_index']
            post_units.append(units.WordHashing(term_index))
        data_pack.apply_on_text(chain_transform(post_units),
                                inplace=True, verbose=verbose)
        return data_pack

    @classmethod
    def _default_units(cls) -> list:
        """Prepare needed process units."""
        return [
            units.Tokenize(),
            units.Lowercase(),
            units.PuncRemoval(),
            units.StopRemoval(),
        ]

    @property
    def with_word_hashing(self):
        """`with_word_hashing` getter."""
        return self._with_word_hashing

    @with_word_hashing.setter
    def with_word_hashing(self, value):
        """`with_word_hashing` setter."""
        self._with_word_hashing = value
