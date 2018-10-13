"""ArcI Preprocessor."""
import os
import errno

import typing
import logging
from tqdm import tqdm

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack
from matchzoo.embedding import Embedding

logger = logging.getLogger(__name__)


class ArcIPreprocessor(engine.BasePreprocessor, preprocessor.SegmentMixin):
    """
    ArcI preprocessor helper.

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

    def __init__(self, fixed_len: list=[32, 32], embedding_file: str=''):
        """Initialization."""
        self._datapack = None
        self._context = {}
        self._embedding_file = embedding_file
        self._fixed_len = fixed_len
        self._vocab_unit = preprocessor.VocabularyUnit()
        self._left_fixedlen_unit = preprocessor.FixedLengthUnit(
                                       self._fixed_len[0])
        self._right_fixedlen_unit = preprocessor.FixedLengthUnit(
                                        self._fixed_len[1])

    def _prepare_stateless_units(self) -> list:
        """Prepare needed process units."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit()
        ]

    def fit(self, inputs: typing.List[tuple]):
        """
        Fit pre-processing context for transformation.

        :param inputs: Inputs to be preprocessed.
        :return: class:`ArcIPreprocessor` instance.
        """
        vocab = []
        units = self._prepare_stateless_units()

        logger.info("Start building vocabulary & fitting parameters.")

        # Convert user input into a datapack object.
        self._datapack = self.segment(inputs, stage='train')

        # Loop through user input to generate words.
        # 1. Used for build vocabulary of words (get dimension).
        # 2. Cached words can be further used to perform input
        #    transformation.
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

        # Initialize a vocabulary process unit to build words vocab.
        self._vocab_unit.fit(vocab)

        if len(self._embedding_file) == 0:
            pass
        elif os.path.isfile(self._embedding_file):
            embed_module = Embedding(embedding_file=self._embedding_file)
            embed_module.build(self._vocab_unit.state['term_index'])
            self._context['embedding_mat'] = embed_module.embedding_mat
        else:
            logger.error("Embedding file [{}] not found."
                         .format(self._embedding_file))
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self._embedding_file)

        # Store the fitted parameters in context.
        self._context['term_index'] = self._vocab_unit.state['term_index']
        self._context['input_shapes'] = [(self._fixed_len[0],),
                                         (self._fixed_len[1],)]
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
        if stage == 'test':
            self._datapack = self.segment(inputs, stage=stage)

        logger.info(f"Start processing input data for {stage} stage.")

        # do preprocessing from scrach.
        units = self._prepare_stateless_units()
        units.append(self._vocab_unit)

        for idx, row in tqdm(self._datapack.left.iterrows()):
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            text = self._left_fixedlen_unit.transform(text)
            self._datapack.left.at[idx, 'text_left'] = text
        for idx, row in tqdm(self._datapack.right.iterrows()):
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            text = self._right_fixedlen_unit.transform(text)
            self._datapack.right.at[idx, 'text_right'] = text

        return self._datapack
