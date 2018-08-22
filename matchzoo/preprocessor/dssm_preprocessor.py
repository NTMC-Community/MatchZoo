"""DSSM Preprocessor."""

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack

import typing
import logging
from tqdm import tqdm

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
        >>> dssm_preprocessor.context['input_shapes'][0][0]
        37
        >>> type(rv_train)
        <class 'matchzoo.datapack.DataPack'>
        >>> context = dssm_preprocessor.context
        >>> dssm_preprocessor_test = DSSMPreprocessor()
        >>> dssm_preprocessor_test.context = context
        >>> test_inputs = [("id0",
        ...                 "id4",
        ...                 "beijing",
        ...                 "I visted beijing yesterday.")]
        >>> rv_test = dssm_preprocessor_test.fit_transform(
        ...     test_inputs,
        ...     stage='test')
        >>> type(rv_test)
        <class 'matchzoo.datapack.DataPack'>

    """

    def __init__(self):
        """Initialization."""
        self._context = {}
        self._datapack = None
        self._cache = []

    @property
    def context(self):
        """Get fitted parameters."""
        return self._context

    @context.setter
    def context(self, context: dict):
        """
        Set pre-fitted context.

        :param context: pre-fitted context.
        """
        self._context = context

    def _prepare_stateless_units(self):
        """Prepare needed process units."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit(),
            preprocessor.NgramLetterUnit()
        ]

    def _build_vocab(
        self,
        inputs: typing.List[tuple]
    ) -> list:
        """
        Build vocabulary before fit transform.

        :param inputs: Use training data as inputs.
        :return vocab: fitted `tri-letters` using
            :meth:`_prepare_stateless_units`.
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
        for idx, row in tqdm(self._datapack.dataframe.iterrows()):
            text = row['text']
            # For each piece of text, apply process unit sequentially.
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)
            # cache tri-letters for transformation.
            self._cache.append((row['id'], text))
        return vocab

    def _check_transoform_state(self, stage: str):
        """Check arguments and context in transformation."""
        if stage not in ['train', 'test']:
            raise ValueError(f'{stage} is not a valid stage name.')
        if not self._context.get('term_index'):
            raise ValueError(
                "Please fit term_index before apply transofm function.")

    def fit(self, inputs: typing.List[tuple]):
        """
        Fit pre-processing context for transformation.

        :param inputs: Inputs to be preprocessed.
        :return: class:`DSSMPreprocessor` instance.
        """
        vocab = self._build_vocab(inputs)

        # Initialize a vocabulary process unit to build tri-letter vocab.
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)

        # Store the fitted parameters in context.
        self._context['term_index'] = vocab_unit.state['term_index']
        dim_triletter = len(vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = [(dim_triletter,), (dim_triletter,)]
        return self

    def transform(
        self,
        inputs: typing.List[tuple],
        stage: str
    ) -> datapack.DataPack:
        """
        Apply trnasformation on data, create `tri-letter` representation.

        :param inputsL Inputs to be preprocessed.
        :param stage: Pre-processing stage, `train` or `test`.

        :return: Transformed data as :class:`DataPack` object.
        """
        outputs = []
        self._check_transoform_state(stage)

        # prepare word hashing unit.
        hashing = preprocessor.WordHashingUnit(
            self._context['term_index'])

        logger.info(f"Start processing input data for {stage} stage.")

        if stage == 'train':
            # use cached data to fit word hashing layer directly.
            for idx, tri_letter in tqdm(self._cache):
                outputs.append((idx, hashing.transform(tri_letter)))

            return self._make_output(output=outputs,
                                     mapping=self._datapack.mapping,
                                     context=self._context,
                                     stage=stage)
        else:
            # do preprocessing from scrach.
            units = self._prepare_stateless_units()
            units.append(hashing)
            self._datapack = self.segmentation(inputs, stage='test')

            for idx, row in tqdm(self._datapack.dataframe.iterrows()):
                text = row['text']
                for unit in units:
                    text = unit.transform(text)
                outputs.append((row['id'], text))

            return self._make_output(output=outputs,
                                     mapping=self._datapack.mapping,
                                     context=self._context,
                                     stage=stage)
