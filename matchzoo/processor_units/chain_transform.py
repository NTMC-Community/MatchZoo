import functools


def chain_transform(units):
    @functools.wraps(chain_transform)
    def wrapper(arg):
        for unit in units:
            arg = unit.transform(arg)
        return arg

    unit_names = ' -> '.join(unit.__class__.__name__ for unit in units)
    wrapper.__name__ += ' of ' + unit_names
    return wrapper
