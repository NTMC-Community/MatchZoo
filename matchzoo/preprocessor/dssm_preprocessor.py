"""DSSM Preprocessor."""

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack


class DSSMPreprocessor(engine.BasePreprocessor):
    """
    DSSM preprocessor helper.

    Example:
        >>> train_inputs = [
        ...     ("beijing", "Beijing is capital of China", 1, ('id0', 'id1')),
        ...     ("beijing", "China is in east Asia", 0, ('id0', 'id2')),
        ...     ("beijing", "Summer in Beijing is hot.", 1, ('id0', 'id3'))
        ... ]
        >>> dssm_preprocessor = DSSMPreprocessor()
        >>> rv_train = dssm_preprocessor.fit_transform(
        ...     train_inputs,
        ...     stage='train')
        >>> dssm_preprocessor.context['dim_triletter']
        25
        >>> type(rv_train)
        <class 'matchzoo.datapack.DataPack'>
        >>> test_inputs = [("beijing",
        ...                 "I visted beijing yesterday.",
        ...                 ('id0', 'id4'))]
        >>> rv_test = dssm_preprocessor.transform(test_inputs, stage='test')
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
        """Set pre-fitted context."""
        self._context = context

    def _prepare_stateless_units(self):
        """Prepare."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit(),
            preprocessor.NgramLetterUnit()
        ]

    def _build_vocab(self, inputs):
        """Build vocabulary before fit transform."""
        vocab = []
        units = self._prepare_stateless_units()
        for input in inputs:
            for unit in units:
                left = unit.transform(input[0])
                right = unit.transform(input[1])
            vocab.extend(left + right)
        return vocab

    def _check_transoform_state(self, stage):
        """check."""
        if stage not in ['train', 'test']:
            msg = f'{stage} is not a valid stage name'
            msg += '`train` or `test` expected.'
            raise ValueError(msg)
        if not self._context.get('term_index'):
            raise ValueError(
                "Please fit term_index before apply transofm function.")

    def fit(self, inputs):
        """Fit parameters."""
        vocab = self._build_vocab(inputs)
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)
        self._context['term_index'] = vocab_unit.state['term_index']
        self._context['dim_triletter'] = len(
            vocab_unit.state['term_index']) + 1
        return self

    def transform(self, inputs, stage):
        """Transform."""
        self._check_transoform_state(stage)
        lst_left, lst_right, lst_labels, lst_ids = ([] for i in range(4))
        units = self._prepare_stateless_units()
        units.append(
            preprocessor.WordHashingUnit(self._context['term_index']))
        for input in inputs:
            for unit in units:
                left = unit.transform(input[0])
                righ = unit.transform(input[1])
            lst_left.append(left)
            lst_right.append(righ)
            if stage == "train":
                lst_labels.append(input[2])
                lst_ids.append(input[3])
            else:
                lst_ids.append(input[2])
        data = {'text_left': lst_left,
                'text_right': lst_right,
                'id': lst_ids}
        if stage == "train":
            data['label'] = lst_labels
        return datapack.DataPack(data=data, context=self._context)
