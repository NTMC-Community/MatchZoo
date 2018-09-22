"""CDSSM Preprocessor."""

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack

import typing
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CDSSMPreprocessor(engine.BasePreprocessor):
    """CDSSM preprocessor helper."""

    @staticmethod
    def _flatten(words: list) -> list:
        """Flatten word list."""
        flattened = list()
        for word in words:
            for letter in word:
                flattened.extend(letter)
        return flattened

    def __init__(self, letter_ngram: int=3,
                 sliding_window: int=3, nb_window: int=5):
        """
        Initialization.

        :param letter_ngram: letter ngram.
        :param sliding_window: sliding window length.
        :param nb_window: window numbers that pads
         different sentences to the same dimensions.
        """
        self._datapack = None
        self._cache_left = []
        self._cache_right = []
        self._context = {}
        self._letter_ngram = letter_ngram
        self._sliding_window = sliding_window
        self._nb_window = nb_window

    def _prepare_stateless_units(self) -> list:
        """Prepare needed process units."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit(),
            preprocessor.NgramLetterUnit(self._letter_ngram),
            preprocessor.SlidingWindowUnit(self._sliding_window,
                                           self._nb_window)
        ]

    def fit(self, inputs: typing.List[tuple]):
        """
        Fit pre-processing context for transformation.

        :param inputs: Inputs to be preprocessed.
        :return: class:`CDSSMPreprocessor` instance.
        """
        vocab = []
        units = self._prepare_stateless_units()

        logger.info("Start building vocabulary & fitting parameters.")

        # Convert user input into a datapack object.
        self._datapack = self.segmentation(inputs, stage='train')

        # Loop through user input to generate tri-letters.
        # 1. Used for build vocabulary of tri-letters (get dimension).
        # 2. Cached tri-letters can be further used to perform input
        #    transformation.
        left = self._datapack.left
        right = self._datapack.right

        for idx, row in tqdm(left.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            vocab.extend(self._flatten(text))
            # cache tri-letters for transformation.
            self._cache_left.append((row.name, text))

        for idx, row in tqdm(right.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            vocab.extend(self._flatten(text))
            # cache tri-letters for transformation.
            self._cache_right.append((row.name, text))

        # Initialize a vocabulary process unit to build tri-letter vocab.
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)

        # Store the fitted parameters in context.
        self._context['term_index'] = vocab_unit.state['term_index']
        dim_triletter = len(vocab_unit.state['term_index']) + 1
        shape = (self._nb_window, self._sliding_window*dim_triletter)
        self._context['input_shapes'] = [shape, shape]
        self._datapack.context = self._context
        return self

    def transform(
        self,
        inputs: typing.List[tuple],
        stage: str
    ) -> datapack.DataPack:
        """
        Apply transformation on data, create `tri-letter` representation.

        :param inputs: Inputs to be preprocessed.
        :param stage: Pre-processing stage, `train` or `test`.

        :return: Transformed data as :class:`DataPack` object.
        """
        if stage not in ['train', 'test']:
            raise ValueError(f'{stage} is not a valid stage name.')
        if not self._context.get('term_index'):
            raise ValueError(
                "Please fit term_index before apply transform function.")

        # prepare word hashing unit.
        hashing = preprocessor.WordHashingWindowUnit(
            self._context['term_index'])

        logger.info(f"Start processing input data for {stage} stage.")

        if stage == 'train':
            # use cached data to fit word hashing layer directly.
            for idx, words in tqdm(self._cache_left):
                words = hashing.transform(words)
                self._datapack.left.at[idx, 'text_left'] = words
            for idx, words in tqdm(self._cache_right):
                words = hashing.transform(words)
                self._datapack.right.at[idx, 'text_right'] = words
            return self._datapack
        else:
            # do preprocessing from scratch.
            units = self._prepare_stateless_units()
            units.append(hashing)
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
