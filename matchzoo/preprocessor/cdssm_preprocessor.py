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

    def __init__(self, sliding_window: int=3, window_num: int=5):
        """Initialization.

        :param sliding_window: sliding window length.
        :param window_num: fixed window number.
        :param pad_value: padding value.
        :param pad_mode: padding mode, `pre` or `post`.
        :param truncate_mode: truncate mode, `pre` or `post`.
        :param remove: user-defined removed tokens.
        """
        self._datapack = None
        self._context = {}
        self._sliding_window = sliding_window
        self._window_num = window_num
        self._length = sliding_window + window_num - 1

    def _prepare_process_units(self) -> list:
        """Prepare needed process units."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit(),
        ]

    def fit(self, inputs: typing.List[tuple]):
        """
        Fit pre-processing context for transformation.

        Can be simplified by compute vocabulary term and index.

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
            text = ngram_unit.transform(text)
            vocab.extend(text)

        for idx, row in tqdm(right.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            text = ngram_unit.transform(text)
            vocab.extend(text)

        # Initialize a vocabulary process unit to build tri-letter vocab.
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)

        # Store the fitted parameters in context.
        self._context['term_index'] = vocab_unit.state['term_index']
        ngram_num = len(vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = [(self._window_num,
                                          ngram_num*self._sliding_window),
                                         (self._window_num,
                                          ngram_num*self._sliding_window)]
        self._datapack.context = self._context
        return self

    def transform(
        self,
        inputs: typing.List[tuple],
        stage: str
    ) -> datapack.DataPack:
        """
        Apply transformation on data, create `letter-trigram` representation.

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
        sliding_unit = preprocessor.SlidingWindowUnit(self._sliding_window)
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
            text = sliding_unit.transform(text)
            if len(text):
                text = np.apply_along_axis(lambda x: np.concatenate(
                    [hash_unit.transform(ngram_unit.transform([w])) for w in x],
                    axis=-1), -1, text)
            self._datapack.left.at[idx, 'text_left'] = text
        for idx, row in tqdm(self._datapack.right.iterrows()):
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            text = sliding_unit.transform(text)
            if len(text):
                text = np.apply_along_axis(lambda x: np.concatenate(
                    [hash_unit.transform(ngram_unit.transform([w])) for w in x],
                    axis=-1), -1, text)
            self._datapack.right.at[idx, 'text_right'] = text

        return self._datapack
