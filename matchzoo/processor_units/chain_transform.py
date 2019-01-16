"""Wrapper function organizes a number of transform functions."""
import typing
import functools

from .processor_units import ProcessorUnit


def chain_transform(units: typing.List[ProcessorUnit]) -> typing.Callable:
    """
    Compose unit transformations into a single function.

    :param units: List of :class:`matchzoo.ProcessorUnit`.
    """
    @functools.wraps(chain_transform)
    def wrapper(arg):
        """Wrapper function of transformations composition."""
        for unit in units:
            arg = unit.transform(arg)
        return arg

    unit_names = ' => '.join(unit.__class__.__name__ for unit in units)
    wrapper.__name__ += ' of ' + unit_names
    return wrapper
