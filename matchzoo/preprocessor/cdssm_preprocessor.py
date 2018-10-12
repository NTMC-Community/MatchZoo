"""CDSSM Preprocessor."""

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack

import typing
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CDSSMPreprocessor(engine.BasePreprocessor, preprocessor.SegmentMixin):
    """CDSSM preprocessor helper.

    Example:
        >>> train_inputs = [
        ...     ("id0", "id1", "beijing", "Beijing is capital of China", 1),
        ...     ("id0", "id2", "beijing", "China is in east Asia", 0),
        ...     ("id0", "id3", "beijing", "Summer in Beijing is hot.", 1)
        ... ]
        >>> cdssm_preprocessor = CDSSMPreprocessor()
        >>> rv_train = cdssm_preprocessor.fit_transform(
        ...     train_inputs,
        ...     stage='train')
        >>> type(rv_train)
        <class 'matchzoo.datapack.DataPack'>
        >>> test_inputs = [("id0",
        ...                 "id4",
        ...                 "beijing",
        ...                 "I visted beijing yesterday.")]
        >>> rv_test = cdssm_preprocessor.fit_transform(
        ...     test_inputs,
        ...     stage='test')
        >>> type(rv_test)
        <class 'matchzoo.datapack.DataPack'>
    """

    def __init__(self, sliding_window: int=3):
        """Initialization.

        :param sliding_window: sliding window length.
        :param remove: user-defined removed tokens.
        """
        self._datapack = None
        self._context = {}
        self._sliding_window = sliding_window

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
        units.append(preprocessor.NgramLetterUnit())

        logger.info("Start building vocabulary & fitting parameters.")

        # Convert user input into a datapack object.
        self._datapack = self.segment(inputs, stage='train')

        for idx, row in tqdm(self._datapack.left.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)

        for idx, row in tqdm(self._datapack.right.iterrows()):
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)

        # Initialize a vocabulary process unit to build tri-letter vocab.
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)

        # Store the fitted parameters in context.
        self._context['term_index'] = vocab_unit.state['term_index']
        dim = len(vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = [(None, dim * self._sliding_window),
                                         (None, dim * self._sliding_window)]
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
        if stage == 'test':
            self._datapack = self.segment(inputs, stage=stage)

        # prepare pipeline unit.
        units = self._prepare_process_units()
        # can not merge into units
        ngram_unit = preprocessor.NgramLetterUnit()
        hash_unit = preprocessor.WordHashingUnit(self._context['term_index'])
        slide_unit = preprocessor.SlidingWindowUnit(self._sliding_window)

        logger.info(f"Start processing input data for {stage} stage.")

        for idx, row in tqdm(self._datapack.left.iterrows()):
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            # apply ngram unit to each token
            text = [ngram_unit.transform([term]) for term in text]
            # apply word hashing to each token ngram.
            text = [hash_unit.transform(term) for term in text]
            # sliding text to user-defined window
            text = slide_unit.transform(text)
            self._datapack.left.at[idx, 'text_left'] = text

        for idx, row in tqdm(self._datapack.right.iterrows()):
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            # apply ngram unit to each token
            text = [ngram_unit.transform([term]) for term in text]
            # apply word hashing to each token ngram.
            text = [hash_unit.transform(term) for term in text]
            # sliding text to user-defined window
            text = slide_unit.transform(text)
            self._datapack.right.at[idx, 'text_right'] = text

        return self._datapack
