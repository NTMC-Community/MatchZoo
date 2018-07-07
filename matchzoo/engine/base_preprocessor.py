"""Base Preprocessor, consist of multiple :class:`ProcessorUnit`.

Each sub-class should employ a sequence of :class:`ProcessorUnit` and
:class:`StatefulProcessorUnit` to handle input data.
"""

import abc
import typing
from matchzoo import preprocessor


class BasePreprocessor(metaclass=abc.ABCMeta):
    """Abstract base class for model-wise processors."""

    @abc.abstractmethod
    def _set_process_units(self):
        """
        Set model-wise process units.

        This method is an abstract method, need to be implemented
        in sub-class.
        """

    @abc.abstractmethod
    def fit_transform(
        self,
        text_left: typing.List[str],
        text_right: typing.List[str],
        labels: typing.List[str]
    ):
        """
        Apply fit-transform on input data.

        This method is an abstract method, need to be implemented
        in sub-class.
        """

    def handle(
        self,
        process_unit: preprocessor.ProcessorUnit,
        input: typing.Any
    ) -> typing.Union[dict, typing.Any]:
        """
        Inference whether a process_unit is `Stateful`.

        :param process_unit: Given a process unit instance.
        :param input: process input text.

        :return ctx: Context as dict, i.e. fitted parameters.
        :return: Transformed user input given transformer.
        """
        ctx = {}
        if isinstance(process_unit, preprocessor.StatefulProcessorUnit):
            process_unit.fit(input)
            ctx = process_unit.state
        return ctx, process_unit.transform(tokens=input)
