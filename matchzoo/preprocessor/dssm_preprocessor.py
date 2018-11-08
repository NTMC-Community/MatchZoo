"""DSSM Preprocessor."""

import typing
import logging

from tqdm import tqdm

from matchzoo import utils
from matchzoo import engine
from matchzoo import preprocessor
from matchzoo.datapack import DataPack

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
        <class 'matchzoo.datapack.DataPack'>
        >>> test_inputs = [("id0",
        ...                 "id4",
        ...                 "beijing",
        ...                 "I visted beijing yesterday.")]
        >>> rv_test = dssm_preprocessor.fit_transform(
        ...     test_inputs,
        ...     stage='predict')
        >>> type(rv_test)
        <class 'matchzoo.datapack.DataPack'>

    """

    def _preprocess_units(self) -> list:
        """Prepare needed process units."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit(),
            preprocessor.NgramLetterUnit(),
        ]

    def fit(self, datapack: DataPack):
        """
        Fit pre-processing context for transformation.

        :param datapack: datapack to be preprocessed.
        :return: class:`DSSMPreprocessor` instance.
        """

        vocab = []
        units = self._preprocess_units()

        def transform_and_extend_vocab(text):
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)

        tqdm.pandas(desc="Preparing `text_left`.")
        datapack.left['text_left'].progress_apply(transform_and_extend_vocab)
        tqdm.pandas(desc="Preparing `text_right`.")
        datapack.right['text_right'].progress_apply(transform_and_extend_vocab)

        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(tqdm(vocab, desc='Fitting vocabulary unit.'))

        self._context['term_index'] = vocab_unit.state['term_index']
        dim_triletter = len(vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = [(dim_triletter,), (dim_triletter,)]

        return self

    @utils.validate_context
    def transform(self, datapack: DataPack) -> DataPack:
        """
        Apply transformation on data, create `tri-letter` representation.

        :param datapack: Inputs to be preprocessed.

        :return: Transformed data as :class:`DataPack` object.
        """
        datapack_copy = datapack.copy()

        term_index = self._context['term_index']
        hashing_unit = preprocessor.WordHashingUnit(term_index)
        units = self._preprocess_units() + [hashing_unit]

        for unit in units:
            unit_name = str(unit.__class__.__name__)

            tqdm.pandas(desc="Processing `text_left` with " + unit_name)
            left = datapack_copy.left['text_left'].progress_apply(
                unit.transform)
            datapack_copy.left['text_left'] = left

            tqdm.pandas(desc="Processing `text_right` with " + unit_name)
            right = datapack_copy.right['text_right'].progress_apply(
                unit.transform)
            datapack_copy.right['text_right'] = right

        return datapack_copy
