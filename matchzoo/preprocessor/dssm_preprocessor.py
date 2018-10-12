"""DSSM Preprocessor."""

import typing
import logging

from tqdm import tqdm

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack

logger = logging.getLogger(__name__)


class DSSMPreprocessor(engine.BasePreprocessor, preprocessor.SegmentMixin):
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
        ...     stage='test')
        >>> type(rv_test)
        <class 'matchzoo.datapack.DataPack'>

    """

    def __init__(self):
        """Initialization."""
        self._datapack = None
        self._context = {}

    def _prepare_process_unit(self) -> list:
        """Prepare needed process units."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit(),
            preprocessor.NgramLetterUnit()
        ]

    def fit(self, inputs: typing.List[tuple]):
        """
        Fit pre-processing context for transformation.

        :param inputs: Inputs to be preprocessed.
        :return: class:`DSSMPreprocessor` instance.
        """
        vocab = []
        units = self._prepare_process_unit()

        logger.info("Start building vocabulary & fitting parameters.")

        # Convert user input into a datapack object.
        self._datapack = self.segment(inputs, stage='train')

        # Loop through user input to generate tri-letters.
        # Used for build vocabulary of tri-letters (get dimension).
        for idx, row in tqdm(self._datapack.left.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)

        for idx, row in tqdm(self._datapack.right.iterrows()):
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
        if stage == 'test':
            self._datapack = self.segment(inputs, stage='test')

        logger.info(f"Start processing input data for {stage} stage.")

        # do preprocessing from scrach.
        units = self._prepare_process_unit()
        # prepare word hashing unit.
        hashing = preprocessor.WordHashingUnit(
            self._context['term_index'])
        units.append(hashing)

        for idx, row in tqdm(self._datapack.left.iterrows()):
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            self._datapack.left.at[idx, 'text_left'] = text
        for idx, row in tqdm(self._datapack.right.iterrows()):
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            self._datapack.right.at[idx, 'text_right'] = text

        return self._datapack
