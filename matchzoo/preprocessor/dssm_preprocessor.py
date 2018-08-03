"""DSSM Preprocessor."""

from matchzoo import engine
from matchzoo import preprocessor
from matchzoo import datapack
from typing import Union, List, Tuple


class DSSMPreprocessor(engine.BasePreprocessor):
    """
    DSSM preprocessor helper.

    Example:
        >>> train_inputs = [
        ...     ("beijing", "Beijing is capital of China", 1),
        ...     ("beijing", "China is in east Asia", 0),
        ...     ("beijing", "Summer in Beijing is hot.", 1)
        ... ]
        >>> dssm_preprocessor = DSSMPreprocessor()
        >>> rv_train = dssm_preprocessor.fit_transform(train_inputs)
        >>> dssm_preprocessor.context['dim_triletter']
        37
        >>> type(rv_train)
        <class 'matchzoo.datapack.DataPack'>
        >>> test_inputs = [("beijing", "I visted beijing yesterday.")]
        >>> rv_test = dssm_preprocessor.transform(test_inputs)
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
        for left, right, label in inputs:
            for unit in units:
                left = unit.transform(left)
                right = unit.transform(right)
            vocab.extend(left + right)
        return vocab

    def fit(self, inputs):
        """Fit parameters."""
        vocab = self._build_vocab(inputs)
        vocab_unit = preprocessor.VocabularyUnit()
        vocab_unit.fit(vocab)
        self._context['term_index'] = vocab_unit.state['term_index']
        self._context['dim_triletter'] = len(
            vocab_unit.state['term_index']) + 1
        return self

    def transform(self, inputs):
        """Transform."""
        output_left = []
        output_righ = []
        if not self._context.get('term_index'):
            raise ValueError(
                "Please fit term_index before apply transofm function")
        inputs, labels = self._detach_labels(inputs)
        units = self._prepare_stateless_units()
        units.append(
            preprocessor.WordHashingUnit(self._context['term_index']))
        for left, righ in inputs:
            for unit in units:
                left = unit.transform(left)
                righ = unit.transform(righ)
            output_left.append(left)
            output_righ.append(righ)
        data = {'text_left': output_left, 'text_right': output_righ}
        if labels:
            data['labels'] = labels
        return datapack.DataPack(data=data, context=self._context)

    def _detach_labels(
        self,
        inputs: list
    ) -> Union[List[Tuple[str, str]], Union[list, None]]:
        """
        Detach labels from inputs.

        During the training and testing phrase, :attr:`inputs`
        usually have different formats. The training phrase
        inputs should be list of triples like [(query, document, label)].
        While during the testing phrase (predict unknown examples),
        the inputs should be list of tuples like [(query, document)].
        This method is used to infer the stage, if `label` exist, detach
        label to a list.

        :param inputs: List of text-left, text-right, label triples.
        :return: Zipped list of text-left and text-right.
        :return: Detached labels, list or None.
        """
        unzipped_inputs = list(zip(*inputs))
        if len(unzipped_inputs) == 3:
            # training stage.
            return zip(unzipped_inputs[0],
                       unzipped_inputs[1]), unzipped_inputs[2]
        else:
            # testing stage.
            return zip(unzipped_inputs[0],
                       unzipped_inputs[1]), None
