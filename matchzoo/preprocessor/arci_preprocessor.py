"""ArcI Preprocessor."""

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack
from matchzoo.embedding import Embedding

import os
import typing
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ArcIPreprocessor(engine.BasePreprocessor):
    """
    ArcI preprocessor helper.

    TODO: NEED REFACTORING.

    Example:
        >>> train_inputs = [
        ...     ("id0", "id1", "beijing", "Beijing is capital of China", 1),
        ...     ("id0", "id2", "beijing", "China is in east Asia", 0),
        ...     ("id0", "id3", "beijing", "Summer in Beijing is hot.", 1)
        ... ]
        >>> arci_preprocessor = ArcIPreprocessor()
        >>> rv_train = arci_preprocessor.fit_transform(
        ...     train_inputs,
        ...     stage='train')
        >>> type(rv_train)
        <class 'matchzoo.datapack.DataPack'>
        >>> test_inputs = [("id0",
        ...                 "id4",
        ...                 "beijing",
        ...                 "I visted beijing yesterday.")]
        >>> rv_test = arci_preprocessor.fit_transform(
        ...     test_inputs,
        ...     stage='test')
        >>> type(rv_test)
        <class 'matchzoo.datapack.DataPack'>

    """

    def __init__(self, embedding_file: str=''):
        """Initialization."""
        self._datapack = None
        self._cache_left = []
        self._cache_right = []
        self._context = {}
        self._embedding_file = embedding_file
        self._vocab_unit = preprocessor.VocabularyUnit()

    def _prepare_stateless_units(self) -> list:
        """Prepare needed process units."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit()
        ]

    def fit(self, inputs: typing.List[tuple]) -> ArcIPreprocessor:
        """
        Fit pre-processing context for transformation.

        :param inputs: Inputs to be preprocessed.
        :return: class:`ArcIPreprocessor` instance.
        """
        vocab = []
        units = self._prepare_stateless_units()

        logger.info("Start building vocabulary & fitting parameters.")

        # Convert user input into a datapack object.
        self._datapack = self.segmentation(inputs, stage='train')

        # Loop through user input to generate words.
        # 1. Used for build vocabulary of words (get dimension).
        # 2. Cached words can be further used to perform input
        #    transformation.
        left = self._datapack.left
        right = self._datapack.right

        for idx, row in tqdm(left.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)
            # cache words for transformation.
            self._cache_left.append((row.name, text))

        for idx, row in tqdm(right.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)
            # cache words for transformation.
            self._cache_right.append((row.name, text))

        # Initialize a vocabulary process unit to build words vocab.
        self._vocab_unit.fit(vocab)

        if os.path.isfile(self._embedding_file):
            embed_module = Embedding(embedding_file=self._embedding_file)
            embed_module.build(self._vocab_unit.state['term_index'])
            self._context['embedding_mat'] = embed_module.embedding_mat
        else:
            logger.info("Embedding file [{}] not found."
                        .format(self._embedding_file))

        # Store the fitted parameters in context.
        self._context['term_index'] = self._vocab_unit.state['term_index']
        self._datapack.context = self._context
        return self

    def transform(
        self,
        inputs: typing.List[tuple],
        stage: str
    ) -> datapack.DataPack:
        """
        Apply transformation on data, create word ids.

        :param inputs: Inputs to be preprocessed.
        :param stage: Pre-processing stage, `train` or `test`.

        :return: Transformed data as :class:`DataPack` object.
        """
        if stage not in ['train', 'test']:
            raise ValueError(f'{stage} is not a valid stage name.')
        if not self._context.get('term_index'):
            raise ValueError(
                "Please fit term_index before apply transofm function.")

        logger.info(f"Start processing input data for {stage} stage.")

        if stage == 'train':
            # use cached data directly.
            for idx, word in tqdm(self._cache_left):
                word = self._vocab_unit.transform(word)
                self._datapack.left.at[idx, 'text_left'] = word
            for idx, word in tqdm(self._cache_right):
                word = self._vocab_unit.transform(word)
                self._datapack.right.at[idx, 'text_right'] = word
            return self._datapack
        else:
            # do preprocessing from scrach.
            units = self._prepare_stateless_units()
            units.append(self._vocab_unit)
            self._datapack = self.segmentation(inputs, stage='test')

            left = self._datapack.left
            right = self._datapack.right

            for idx, row in tqdm(left.iterrows()):
                text = row.text_left
                for unit in units:
                    text = unit.transform(text)
                self._datapack.left.at[idx, 'text_left'] = text
            for idx, row in tqdm(right.iterrows()):
                text = row.text_right
                for unit in units:
                    text = unit.transform(text)
                self._datapack.right.at[idx, 'text_right'] = text

            return self._datapack
