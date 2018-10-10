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

    def __init__(self, sliding_window: int=3, window_num: int=5,
                 pad_value: str='UNKNOW', pad_mode: str='pre',
                 truncate_mode: str='pre', remove: list=[]):
        """Initialization."""
        self._datapack = None
        if truncate_mode not in ['pre', 'post']:
            raise ValueError('{} is not a vaild '
                             'truncate mode.'.format(truncate_mode))
        if pad_mode not in ['pre', 'post']:
            raise ValueError('{} is not a vaild '
                             'pad mode.'.format(pad_mode))
        self._context = {}
        self._sliding_window = sliding_window
        self._window_num = window_num
        self._length = sliding_window + window_num - 1
        self._pad_value = pad_value
        self._pad_mode = pad_mode
        self._truncate_mode = truncate_mode
        if pad_value not in remove:
            remove.append(pad_value)
        self._remove = remove

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

        :param inputs: Inputs to be preprocessed.
        :return: class:`CDSSMPreprocessor` instance.
        """
        vocab = []
        units = self._prepare_process_units()
        ngram_unit = preprocessor.NgramLetterUnit(self._remove)

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
        vocab_unit = preprocessor.VocabularyUnit(self._remove)
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
        sliding_unit = preprocessor.SlidingWindowUnit(self._sliding_window)
        ngram_unit = preprocessor.NgramLetterUnit(self._remove)
        hash_unit = preprocessor.WordHashingUnit(
            self._context['term_index'])
        self._datapack = self.segment(inputs, stage=stage)
        self._datapack.context = self._context

        logger.info(f"Start processing input data for {stage} stage.")

        for idx, row in tqdm(self._datapack.left.iterrows()):
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            fixed = np.full([self._length], self._pad_value)
            if self._truncate_mode == 'pre':
                trunc_text = text[-self._length:]
            else:
                trunc_text = text[:self._length]
            if self._pad_mode == 'post':
                fixed[:len(trunc_text)] = trunc_text
            else:
                fixed[-len(trunc_text):] = trunc_text
            text = sliding_unit.transform(fixed)
            text = np.apply_along_axis(lambda x: np.concatenate(
                [hash_unit.transform(ngram_unit.transform([w])) for w in x],
                axis=-1), -1, text)
            self._datapack.left.at[idx, 'text_left'] = text
        for idx, row in tqdm(self._datapack.right.iterrows()):
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            fixed = np.full([self._length], self._pad_value)
            if self._truncate_mode == 'pre':
                trunc_text = text[-self._length:]
            else:
                trunc_text = text[:self._length]
            if self._pad_mode == 'post':
                fixed[:len(trunc_text)] = trunc_text
            else:
                fixed[-len(trunc_text):] = trunc_text
            text = sliding_unit.transform(fixed)
            text = np.apply_along_axis(lambda x: np.concatenate(
                [hash_unit.transform(ngram_unit.transform([w])) for w in x],
                axis=-1), -1, text)
            self._datapack.right.at[idx, 'text_right'] = text

        return self._datapack
