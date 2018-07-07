"""Pipeline definition, consist of multiple :class:`ProcessorUnit`.

Each model-wise preprocessor should employ a sequence of :class:`ProcessorUnit`
and :class:`StatefulProcessorUnit` to handle input data.
"""

import typing

from matchzoo import preprocessor


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
        >>> input = 'Test sentence to be cleaned.'
        >>> state, rv = pipe.fit_transform(input)
        >>> print(state)
        {'input_dim': 24}
        >>> 'ent' in rv
        True
        >>> len(pipe.processor_units)
        2

    """

    def __init__(self):
        """Class Initialization."""
        self._processor_units = []
        self.state = {}
        self.rv = None

    def fit_transform(self, input: typing.Any):
        """
        Apply fit-transform on input data.

        :param input: Input data to the :class:`Pipeline`.

        :return state: Fitted parameters based on processor units.
        :return rv: Transformed returned value.
        """
        for idx, unit in enumerate(self._processor_units):
            if idx == 0:
                # Handle input using first processor unit.
                ctx, self.rv = self._handle(unit, input)
            else:
                # Handle return value using rest processor units.
                ctx, self.rv = self._handle(unit, self.rv)
            self.state.update(ctx)
        return self.state, self.rv

    def add(self, processor_unit: preprocessor.ProcessorUnit):
        """
        Add processor unit to :class:`Pipeline`.

        :param: `processor unit` to be added.
        """
        # Remove duplicate.
        if processor_unit in self._processor_units:
            raise ValueError(
                "Trying to add exist processor unit!"
            )
        self._processor_units.append(processor_unit)

    def remove(self, processor_unit: preprocessor.ProcessorUnit):
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
    def processor_units(self) -> typing.List[preprocessor.ProcessorUnit]:
        """
        Get processor units within :class:`Pipeline`.

        :return: `processor units` within :class:`Pipeline`.
        """
        return self._processor_units

    def _handle(
        self,
        processor_unit: preprocessor.ProcessorUnit,
        input: typing.Any
    ) -> typing.Union[dict, typing.Any]:
        """
        Handle input data with current `processor unit`.

        Inference whether a `processor_unit` is `Stateful`.

        :param processor_unit: Given a processor unit instance.
        :param input: Input text to be processed.

        :return ctx: Context as dict, i.e. fitted parameters.
        :return: Transformed user input given transformer.
        """
        ctx = {}
        if isinstance(processor_unit, preprocessor.StatefulProcessorUnit):
            processor_unit.fit(input)
            ctx = processor_unit.state
        return ctx, processor_unit.transform(input)
