"""
Hyper parameter search spaces wrapping `hyperopt`.

See `hyperopt`'s documentation for more details:
https://github.com/hyperopt/hyperopt/wiki/FMin

Reason of these wrappers:

    A hyper space in `hyperopt` requires a `label` to instantiate. This `label`
    is used later as a reference to original hyper space that is sampled. In
    `matchzoo`, hyper spaces are used in :class:`matchzoo.engine.Param`s. Only
    if a hyper space's label matches its parent
    :class:`matchzoo.engine.Param`'s name, `matchzoo` can correctly
    back-refrenced the parameter got sampled. This can be done by asking the
    user always use the same name for a parameter and its hyper space, but
    typos can occur. As a result, these wrappers are created to hide hyper
    spaces' `label`s, and always correctly bind them with its parameter's
    name.

Example:

    >>> from hyperopt.pyll.stochastic import sample
    >>> numbers = [0, 1, 2]
    >>> sample(choice(options=numbers)('numbers')) in numbers
    True
    >>> 0 <= sample(quniform(low=0, high=9)('digit')) <= 9
    True
    >>> 0 <= sample(uniform(low=0, high=9)('digit')) <= 9
    True

"""

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

# TODO: more API
