import functools

from hyperopt import hp


def _hyperopt_api(matchzoo_func):
    @functools.wraps(matchzoo_func)
    def wrapper(*args, **kwargs):
        partial_hyperopt_func = matchzoo_func(*args, **kwargs)
        partial_hyperopt_func.is_hyperopt_api = True
        return partial_hyperopt_func

    return wrapper


@_hyperopt_api
def choice(options) -> functools.partial:
    return functools.partial(hp.choice, options=options)


@_hyperopt_api
def uniform(low, high) -> functools.partial:
    return functools.partial(hp.uniform, low=low, high=high)


@_hyperopt_api
def quniform(low, high, q=1) -> functools.partial:
    return functools.partial(hp.quniform, low=low, high=high, q=q)
