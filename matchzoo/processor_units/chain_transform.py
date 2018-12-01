"""Wrapper function organizes a number of transform functions."""
import functools


def chain_transform(units):
    """Compose unit transformations into a single function."""
    @functools.wraps(chain_transform)
    def wrapper(arg):
        """Wrapper function of transformations composition."""
        for unit in units:
            arg = unit.transform(arg)
        return arg

    unit_names = ' => '.join(unit.__class__.__name__ for unit in units)
    wrapper.__name__ += ' of ' + unit_names
    return wrapper
