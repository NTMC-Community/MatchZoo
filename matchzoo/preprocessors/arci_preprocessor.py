"""ArcI Preprocessor."""
import errno
import logging
import os
import typing

from tqdm import tqdm

from matchzoo import DataPack, pack
from matchzoo import engine
from matchzoo import preprocessors
from matchzoo.embedding import Embedding

logger = logging.getLogger(__name__)


class ArcIPreprocessor(engine.BasePreprocessor):
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
        <class 'matchzoo.data_pack.DataPack'>
        >>> test_inputs = [("id0",
        ...                 "id4",
        ...                 "beijing",
        ...                 "I visted beijing yesterday.")]
        >>> rv_test = arci_preprocessor.fit_transform(
        ...     test_inputs,
        ...     stage='predict')
        >>> type(rv_test)
        <class 'matchzoo.data_pack.DataPack'>

    """

    def __init__(self,
                 fixed_length: list = [32, 32],
                 embedding_file: str = ''):
        """Initialization."""
        self.data_pack = None
        self._embedding_file = embedding_file
        self._fixed_length = fixed_length
        self._vocab_unit = preprocessors.VocabularyUnit()
        self._left_fixedlen_unit = preprocessors.FixedLengthUnit(
            self._fixed_length[0])
        self._right_fixedlen_unit = preprocessors.FixedLengthUnit(
            self._fixed_length[1])

    def _prepare_stateless_units(self) -> list:
        """Prepare needed process units."""
        return [
            preprocessors.TokenizeUnit(),
            preprocessors.LowercaseUnit(),
            preprocessors.PuncRemovalUnit(),
            preprocessors.StopRemovalUnit()
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

        # Convert user input into a data_pack object.
        self.data_pack = pack(inputs, stage='train')

        # Loop through user input to generate words.
        # 1. Used for build vocabulary of words (get dimension).
        # 2. Cached words can be further used to perform input
        #    transformation.
        for idx, row in tqdm(self.data_pack.left.iterrows()):
            # For each piece of text, apply process unit sequentially.
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)

        for idx, row in tqdm(self.data_pack.right.iterrows()):
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
            self.data_pack.context['embedding_mat'] = embed_module.embedding_mat
        else:
            logger.error("Embedding file [{}] not found."
                         .format(self._embedding_file))
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    self._embedding_file)

        # Store the fitted parameters in context.
        self.data_pack.context['term_index'] = self._vocab_unit.state[
            'term_index']
        self.data_pack.context['input_shapes'] = [(self._fixed_length[0],),
                                                 (self._fixed_length[1],)]
        return self

    @engine.validate_context
    def transform(
        self,
        inputs: typing.List[tuple],
        stage: str
    ) -> DataPack:
        """
        Apply transformation on data, create word ids.

        :param inputs: Inputs to be preprocessed.
        :param stage: Pre-processing stage, `train`, `evaluate`, or `predict`.

        :return: Transformed data as :class:`DataPack` object.
        """
        if stage in ['evaluate', 'predict']:
            self.data_pack = pack(inputs, stage=stage)

        logger.info(f"Start processing input data for {stage} stage.")

        # do preprocessing from scrach.
        units = self._prepare_stateless_units()
        units.append(self._vocab_unit)

        for idx, row in tqdm(self.data_pack.left.iterrows()):
            text = row.text_left
            for unit in units:
                text = unit.transform(text)
            length = len(text)
            text = self._left_fixedlen_unit.transform(text)
            length = min(length, self._fixed_length[0])
            self.data_pack.left.at[idx, 'text_left'] = text
            self.data_pack.left.at[idx, 'length_left'] = length
        for idx, row in tqdm(self.data_pack.right.iterrows()):
            text = row.text_right
            for unit in units:
                text = unit.transform(text)
            length = len(text)
            text = self._right_fixedlen_unit.transform(text)
            length = min(length, self._fixed_length[1])
            self.data_pack.right.at[idx, 'text_right'] = text
            self.data_pack.right.at[idx, 'length_right'] = length

        return self.data_pack
