"""DSSM Preprocessor."""

import typing
import logging

from tqdm import tqdm

from matchzoo import utils
from matchzoo import engine
from matchzoo import preprocessor
from matchzoo.datapack import DataPack

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _default_preprocess_units() -> list:
        """Prepare needed process units."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit(),
            preprocessor.NgramLetterUnit()
        ]

    def fit(self, datapack: DataPack):
        """
        Fit pre-processing context for transformation.

        :param datapack: datapack to be preprocessed.
        :return: class:`DSSMPreprocessor` instance.
        """
        vocab = []
        units = self._default_preprocess_units()

        logger.info("Start building vocabulary & fitting parameters.")

        # Loop through user input to generate tri-letters.
        # Used for build vocabulary of tri-letters (get dimension).
        for idx, row in tqdm(datapack.left.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)

        for idx, row in tqdm(datapack.right.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)

        # Initialize a vocabulary process unit to build tri-letter vocab.
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)

        # Store the fitted parameters in context.
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

        units = self._default_preprocess_units()
        hashing = preprocessor.WordHashingUnit(self._context['term_index'])
        units.append(hashing)

        for idx, row in tqdm(datapack_copy.left.iterrows()):
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            datapack_copy.left.at[idx, 'text_left'] = text
        for idx, row in tqdm(datapack_copy.right.iterrows()):
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            datapack_copy.right.at[idx, 'text_right'] = text

        return datapack_copy
