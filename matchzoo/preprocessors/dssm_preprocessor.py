"""DSSM Preprocessor."""

from tqdm import tqdm

from matchzoo.data_pack import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from .chain_transform import chain_transform
from .build_vocab_unit import build_vocab_unit
from . import units

tqdm.pandas()


class DSSMPreprocessor(BasePreprocessor):
    """DSSM Model preprocessor."""

    def __init__(self, with_word_hashing: bool = True):
        """
        DSSM Model preprocessor.

        The word hashing step could eats up a lot of memory. To workaround
        this problem, set `with_word_hashing` to `False` and use  a
        :class:`matchzoo.DynamicDataGenerator` with a
        :class:`matchzoo.preprocessor.units.WordHashing`.

        :param with_word_hashing: Include a word hashing step if `True`.

        Example:
            >>> import matchzoo as mz
            >>> train_data = mz.datasets.toy.load_data()
            >>> test_data = mz.datasets.toy.load_data(stage='test')
            >>> dssm_preprocessor = mz.preprocessors.DSSMPreprocessor()
            >>> train_data_processed = dssm_preprocessor.fit_transform(
            ...     train_data, verbose=0
            ... )
            >>> type(train_data_processed)
            <class 'matchzoo.data_pack.data_pack.DataPack'>
            >>> test_data_transformed = dssm_preprocessor.transform(test_data,
            ...                                                     verbose=0)
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

        func = chain_transform(self._default_units())
        data_pack = data_pack.apply_on_text(func, verbose=verbose)
        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)

        self._context['vocab_unit'] = vocab_unit
        vocab_size = len(vocab_unit.state['term_index'])
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size
        self._context['input_shapes'] = [(vocab_size,), (vocab_size,)]
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create `tri-letter` representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        units_ = self._default_units()
        if self._with_word_hashing:
            term_index = self._context['vocab_unit'].state['term_index']
            units_.append(units.WordHashing(term_index))
        func = chain_transform(units_)
        data_pack.apply_on_text(func, inplace=True, verbose=verbose)
        return data_pack

    @classmethod
    def _default_units(cls) -> list:
        """Prepare needed process units."""
        return [
            units.Tokenize(),
            units.Lowercase(),
            units.PuncRemoval(),
            units.StopRemoval(),
            units.NgramLetter(),
        ]

    @property
    def with_word_hashing(self):
        """`with_word_hashing` getter."""
        return self._with_word_hashing

    @with_word_hashing.setter
    def with_word_hashing(self, value):
        """`with_word_hashing` setter."""
        self._with_word_hashing = value
