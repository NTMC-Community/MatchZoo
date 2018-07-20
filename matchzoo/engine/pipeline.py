"""Pipeline definition, consist of multiple :class:`ProcessorUnit`.

Each model-wise preprocessor should employ a sequence of :class:`ProcessorUnit`
and :class:`StatefulProcessorUnit` to handle input data.
"""

import abc
import typing

from matchzoo import preprocessor
from matchzoo import datapack


class Pipeline(object):
    """Pipeline to handle data given processor units.

    Example:
        >>> pipe = Pipeline()
        >>> tu = preprocessor.TokenizeUnit()
        >>> lu = preprocessor.LowercaseUnit()
        >>> nu = preprocessor.NgramLetterUnit()
        >>> pipe.add(tu)
        >>> pipe.add(lu)
        >>> pipe.add(nu)
        >>> print(len(pipe))
        3
        >>> pipe.remove(lu)
        >>> print(len(pipe))
        2
        >>> len(pipe.processor_units)
        2

    """

    def __init__(self):
        """Class Initialization."""
        self._processor_units = []
        self.context = {}
        self.dataframe = None

    def fit(self, input: typing.Any) -> typing.Callable:
        """Fit."""
        self.context = self._fit(input)
        return self

    def transform(self, input) -> datapack.DataPack:
        """Transform."""
        self.dataframe = self._transform(input)
        return datapack.DataPack(self.dataframe, self.context)

    def fit_transform(self, input) -> datapack.DataPack:
        """Fit transform."""
        return self.fit(input).transform(input)

    @abc.abstractmethod
    def _fit(self, input: typing.Any):
        """Fit."""

    @abc.abstractmethod
    def _transform(self, input):
        """Transform."""

    def add(
        self,
        processor_unit: preprocessor.ProcessorUnit
    ):
        """
        Add processor unit to :class:`Pipeline`.

        :param: `processor unit` to be added.
        """
        # Remove duplicate.
        if processor_unit in self._processor_units:
            raise ValueError(
                "Trying to add existed processor unit!"
            )
        self._processor_units.append(processor_unit)

    def remove(
        self,
        processor_unit: preprocessor.ProcessorUnit
    ):
        """
        Remove process unit from :class:`Pipeline`.

        :param processor_unit: `processor unit` to be removed.
        """
        if processor_unit not in self._processor_units:
            raise ValueError(
                "Trying to remove a non-exist processor unit!"
            )
        self._processor_units.remove(processor_unit)

    def __len__(self) -> int:
        """Get number of processor units."""
        return len(self._processor_units)

    @property
    def processor_units(
        self
    ) -> typing.List[preprocessor.ProcessorUnit]:
        """
        Get processor units within :class:`Pipeline`.

        :return: `processor units` within :class:`Pipeline`.
        """
        return self._processor_units
