"""DSSM Preprocessor."""

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack

import typing


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
        >>> dssm_preprocessor.context['dim_triletter']
        25
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
        for _, _, text_left, text_right, _ in inputs:
            for unit in units:
                left = unit.transform(text_left)
                right = unit.transform(text_right)
            vocab.extend(left + right)
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
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)
        self._context['term_index'] = vocab_unit.state['term_index']
        self._context['dim_triletter'] = len(
            vocab_unit.state['term_index']) + 1
        return self

    def transform(
        self,
        inputs: typing.List[tuple],
        stage: str
    ):
        """
        Apply trnasformation on data, create `tri-letter` representation.

        :param inputsL Inputs to be preprocessed.
        :param stage: Pre-processing stage, `train` or `test`.

        :return: Transformed data as :class:`DataPack` object.
        """
        outputs = []
        self._check_transoform_state(stage)
        units = self._prepare_stateless_units()
        units.append(
            preprocessor.WordHashingUnit(self._context['term_index']))
        for input in inputs:
            for unit in units:
                left = unit.transform(input[2])
                right = unit.transform(input[3])
            if stage == 'train':
                outputs.append((input[0], input[1], left, right, input[4]))
            else:
                outputs.append((input[0], input[1], left, right))
        column_names = ['id_left', 'id_right', 'text_left', 'text_right']
        if stage == 'train':
            column_names.append('label')
        return datapack.DataPack(
            data=outputs,
            context=self._context,
            columns=column_names)
