""":class:`BasePreprocessor` define input and ouutput for processors."""

import abc
from typing import Union, List, Tuple
from matchzoo import datapack


class BasePreprocessor(metaclass=abc.ABCMeta):
    """:class:`BasePreprocessor` to input handle data."""

    @abc.abstractmethod
    def fit(self, inputs: list) -> 'BasePreprocessor':
        """
        Fit parameters on input data.

        This method is an abstract base method, need to be
        implemented in the child class.

        This method is expected to return itself as a callable
        object.

        :param inputs: List of text-left, text-right, label triples.
        """

    @abc.abstractmethod
    def transform(self, inputs: list) -> datapack.DataPack:
        """
        Transform input data to expected manner.

        This method is an abstract base method, need to be
        implemented in the child class.

        :param inputs: List of text-left, text-right, label triples,
            or list of text-left, text-right tuples (test stage).
        """

    def fit_transform(self, inputs: list) -> datapack.DataPack:
        """
        Call fit-transform.

        :param inputs: List of text-left, text-right, label triples.
        """
        return self.fit(inputs).transform(inputs)

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
