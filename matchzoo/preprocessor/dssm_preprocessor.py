"""DSSM Preprocessor."""

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack


class DSSMPreprocessor(engine.BasePreprocessor):
    """
    DSSM preprocessor helper.

    Example:
        >>> train_inputs = [
        ...     ("beijing", "Beijing is capital of China", 1),
        ...     ("beijing", "China is in east Asia", 0),
        ...     ("beijing", "Summer in Beijing is hot.", 1)
        ... ]
        >>> dssm_preprocessor_train = DSSMPreprocessor(train_inputs)
        >>> rv_train = dssm_preprocessor_train.fit_transform()
        >>> type(rv_train)
        <class 'matchzoo.datapack.DataPack'>
        >>> fitted_parameters = rv_train.context
        >>> test_inputs = [("beijing", "I visited Beijing.", 1)]
        >>> dssm_preprocessor_test = DSSMPreprocessor(
        ...     test_inputs,
        ...     fitted_parameters)
        >>> rv_test = dssm_preprocessor_test.transform()
        >>> type(rv_test)
        <class 'matchzoo.datapack.DataPack'>

    """

    def __init__(self, inputs, context={}):
        """Initialization."""
        self.inputs = inputs
        self.context = context

    def _prepare_stateless_units(self):
        """Prepare."""
        return [
            preprocessor.TokenizeUnit(),
            preprocessor.LowercaseUnit(),
            preprocessor.PuncRemovalUnit(),
            preprocessor.StopRemovalUnit(),
            preprocessor.NgramLetterUnit()
        ]

    def _build_vocab(self):
        """Build vocabulary before fit transform."""
        vocab = []
        units = self._prepare_stateless_units()
        for left, right, label in self.inputs:
            for unit in units:
                left = unit.transform(left)
                right = unit.transform(right)
            # Extend tri-letters into vocab.
            vocab.extend(list(left.union(right)))
        return vocab

    def fit(self):
        """Fit parameters."""
        vocab = self._build_vocab()
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)
        self.context['term_index'] = vocab_unit.state['term_index']
        self.context['dim_triletter'] = len(vocab_unit.state['term_index']) + 1
        return self

    def transform(self):
        """Transform."""
        output = []
        units = self._prepare_stateless_units()
        term_index = self.context.get('term_index', None)
        if not term_index:
            raise ValueError(
                "Before apply transofm function, please fit term_index first!")
        units.append(preprocessor.WordHashingUnit(term_index))
        for left, right, label in self.inputs:
            for unit in units:
                left = unit.transform(left)
                right = unit.transform(right)
            output.append([left, right, label])
        return datapack.DataPack(output, self.context)
