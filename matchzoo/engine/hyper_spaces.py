"""Hyper parameter search spaces wrapping `hyperopt`."""

import typing
import numbers

import hyperopt
import hyperopt.pyll.base


class HyperoptProxy(object):
    """
    Hyperopt proxy class.

    See `hyperopt`'s documentation for more details:
    https://github.com/hyperopt/hyperopt/wiki/FMin

    Reason of these wrappers:

        A hyper space in `hyperopt` requires a `label` to instantiate. This
        `label` is used later as a reference to original hyper space that is
        sampled. In `matchzoo`, hyper spaces are used in
        :class:`matchzoo.engine.Param`. Only if a hyper space's label
        matches its parent :class:`matchzoo.engine.Param`'s name, `matchzoo`
        can correctly back-refrenced the parameter got sampled. This can be
        done by asking the user always use the same name for a parameter and
        its hyper space, but typos can occur. As a result, these wrappers
        are created to hide hyper spaces' `label`, and always correctly
        bind them with its parameter's name.

    Example:

        >>> from hyperopt.pyll.stochastic import sample
        >>> numbers = [0, 1, 2]
        >>> sample(choice(options=numbers)('numbers')) in numbers
        True
        >>> 0 <= sample(quniform(low=0, high=9)('digit')) <= 9
        True

    """

    def __init__(
            self,
            hyperopt_func: typing.Callable[..., hyperopt.pyll.Apply],
            **kwargs
    ):
        """
        :class:`HyperoptProxy` constructor.

        :param hyperopt_func: Target :module:`hyperopt.hp` function to proxy.
        :param kwargs: Keyword arguments of the proxy function, must pass all
            parameters in `hyperopt_func`.
        """
        self._func = hyperopt_func
        self._kwargs = kwargs

    def __call__(self, name: str) -> hyperopt.pyll.Apply:
        """
        Attach `name` as :module:`hyperopt.hp`'s `label`.

        :param name:
        :return: a :module:`hyperopt` ready search space
        """
        return self._func(name, **self._kwargs)


class choice(HyperoptProxy):
    """:func:`hyperopt.hp.choice` proxy."""

    def __init__(self, options: list):
        """
        :func:`hyperopt.hp.choice` proxy.

        :param options: options to search from
        """
        super().__init__(hyperopt_func=hyperopt.hp.choice, options=options)


class quniform(HyperoptProxy):
    """:func:`hyperopt.hp.quniform` proxy."""

    def __init__(
            self,
            low: numbers.Number,
            high: numbers.Number,
            q: numbers.Number = 1
    ):
        """
        :func:`hyperopt.hp.quniform` proxy.

        :param low: lower bound of the space
        :param high: upper bound of the space
        :param q: similar to the `step` in the python built-in `range`
        """
        super().__init__(hyperopt_func=hyperopt.hp.quniform,
                         low=low,
                         high=high, q=q)


class uniform(HyperoptProxy):
    """:func:`hyperopt.hp.uniform` proxy."""

    def __init__(
            self,
            low: numbers.Number,
            high: numbers.Number
    ):
        """
        :func:`hyperopt.hp.uniform` proxy.

        :param low: lower bound of the space
        :param high: upper bound of the space
        """
        super().__init__(hyperopt_func=hyperopt.hp.uniform, low=low, high=high)
