"""Wrapper function organizes a number of transform functions."""
import functools


def chain_transform(units):
    """Organize a chain of transform functions."""
    @functools.wraps(chain_transform)
    def wrapper(arg):
        """Execute the transform function sequentially."""
        for unit in units:
            arg = unit.transform(arg)
        return arg

    unit_names = ' => '.join(unit.__class__.__name__ for unit in units)
    wrapper.__name__ += ' of ' + unit_names
    return wrapper
