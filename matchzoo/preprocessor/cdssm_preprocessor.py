"""DSSM Preprocessor."""

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack

import typing
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CDSSMPreprocessor(engine.BasePreprocessor, preprocessor.SegmentMixin):
    """CDSSM preprocessor helper."""

    def __init__(self, sliding_window: int=3):
        """Initialization."""
        self._datapack = None
        # self._cache_left = []
        # self._cache_right = []
        self._context = {}
        self._sliding_window = sliding_window

    def _prepare_process_units(self) -> list:
        """Prepare needed process units."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit(),
            preprocessor.SlidingWindowUnit(self._sliding_window),
        ]

    def fit(self, inputs: typing.List[tuple]):
        """
        Fit pre-processing context for transformation.

        :param inputs: Inputs to be preprocessed.
        :return: class:`CDSSMPreprocessor` instance.
        """
        vocab = []
        units = self._prepare_process_units()
        ngram_unit = preprocessor.NgramLetterUnit()

        logger.info("Start building vocabulary & fitting parameters.")

        # Convert user input into a datapack object.
        self._datapack = self.segment(inputs, stage='train')

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
            tmp = []
            for window in text:
                tmp.extend(ngram_unit.transform(window))
            # text = np.apply_along_axis(lambda x: ngram_unit.transform(x), -1, text)
            vocab.extend(tmp)

        for idx, row in tqdm(right.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            tmp = []
            for window in text:
                tmp.extend(ngram_unit.transform(window))
            # text = np.apply_along_axis(lambda x: ngram_unit.transform(x), -1, text)
            vocab.extend(tmp)

        # Initialize a vocabulary process unit to build tri-letter vocab.
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)

        # Store the fitted parameters in context.
        self._context['term_index'] = vocab_unit.state['term_index']
        dim_triletter = len(vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = [(None, dim_triletter*self._sliding_window),
                                         (None, dim_triletter*self._sliding_window)]
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
                "Please fit term_index before apply transofm function.")

        # prepare pipeline unit.
        units = self._prepare_process_units()
        ngram_unit = preprocessor.NgramLetterUnit()
        hash_unit = preprocessor.WordHashingUnit(
            self._context['term_index'])
        self._datapack = self.segment(inputs, stage=stage)
        self._datapack.context = self._context

        logger.info(f"Start processing input data for {stage} stage.")

        for idx, row in tqdm(self._datapack.left.iterrows()):
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            tmp = []
            for window in text:
                out = []
                for word in window:
                    out.extend(hash_unit.transform(ngram_unit.transform(word)))
                tmp.append(out)
            self._datapack.left.at[idx, 'text_left'] = tmp
        for idx, row in tqdm(self._datapack.right.iterrows()):
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            tmp = []
            for window in text:
                out = []
                for word in window:
                    out.extend(hash_unit.transform(ngram_unit.transform(word)))
                tmp.append(out)
            self._datapack.left.at[idx, 'text_left'] = tmp

        return self._datapack
