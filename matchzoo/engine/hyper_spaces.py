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

    Examples::
        >>> import matchzoo as mz
        >>> from hyperopt.pyll.stochastic import sample

    Basic Usage:
        >>> model = mz.models.DenseBaseline()
        >>> sample(model.params.hyper_space)  # doctest: +SKIP
         {'mlp_num_layers': 1.0, 'mlp_num_units': 274.0}

    Arithmetic Operations:
        >>> new_space = 2 ** mz.hyper_spaces.quniform(2, 6)
        >>> model.params.get('mlp_num_layers').hyper_space = new_space
        >>> sample(model.params.hyper_space)  # doctest: +SKIP
        {'mlp_num_layers': 8.0, 'mlp_num_units': 292.0}

    """

    def __init__(
        self,
        hyperopt_func: typing.Callable[..., hyperopt.pyll.Apply],
        **kwargs
    ):
        """
        :class:`HyperoptProxy` constructor.

        :param hyperopt_func: Target `hyperopt.hp` function to proxy.
        :param kwargs: Keyword arguments of the proxy function, must pass all
            parameters in `hyperopt_func`.
        """
        self._func = hyperopt_func
        self._kwargs = kwargs

    def convert(self, name: str) -> hyperopt.pyll.Apply:
        """
        Attach `name` as `hyperopt.hp`'s `label`.

        :param name:
        :return: a `hyperopt` ready search space
        """
        return self._func(name, **self._kwargs)

    def __add__(self, other):
        """__add__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x + y)

    def __radd__(self, other):
        """__radd__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x + y)

    def __sub__(self, other):
        """__sub__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x - y)

    def __rsub__(self, other):
        """__rsub__."""
        return _wrap_as_composite_func(self, other, lambda x, y: y - x)

    def __mul__(self, other):
        """__mul__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x * y)

    def __rmul__(self, other):
        """__rmul__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x * y)

    def __truediv__(self, other):
        """__truediv__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        """__rtruediv__."""
        return _wrap_as_composite_func(self, other, lambda x, y: y / x)

    def __floordiv__(self, other):
        """__floordiv__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x // y)

    def __rfloordiv__(self, other):
        """__rfloordiv__."""
        return _wrap_as_composite_func(self, other, lambda x, y: y // x)

    def __pow__(self, other):
        """__pow__."""
        return _wrap_as_composite_func(self, other, lambda x, y: x ** y)

    def __rpow__(self, other):
        """__rpow__."""
        return _wrap_as_composite_func(self, other, lambda x, y: y ** x)

    def __neg__(self):
        """__neg__."""
        return _wrap_as_composite_func(self, None, lambda x, _: -x)


def _wrap_as_composite_func(self, other, func):
    def _wrapper(name, **kwargs):
        return func(self._func(name, **kwargs), other)

    return HyperoptProxy(_wrapper, **self._kwargs)


class choice(HyperoptProxy):
    """:func:`hyperopt.hp.choice` proxy."""

    def __init__(self, options: list):
        """
        :func:`hyperopt.hp.choice` proxy.

        :param options: options to search from
        """
        super().__init__(hyperopt_func=hyperopt.hp.choice, options=options)
        self._options = options

    def __str__(self):
        """:return: `str` representation of the hyper space."""
        return f'choice in {self._options}'


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

        If using with integer values, then `high` is exclusive.

        :param low: lower bound of the space
        :param high: upper bound of the space
        :param q: similar to the `step` in the python built-in `range`
        """
        super().__init__(hyperopt_func=hyperopt.hp.quniform,
                         low=low,
                         high=high, q=q)
        self._low = low
        self._high = high
        self._q = q

    def __str__(self):
        """:return: `str` representation of the hyper space."""
        return f'quantitative uniform distribution in  ' \
               f'[{self._low}, {self._high}), with a step size of {self._q}'


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
        self._low = low
        self._high = high

    def __str__(self):
        """:return: `str` representation of the hyper space."""
        return f'uniform distribution in  [{self._low}, {self._high})'


def sample(space):
    """
    Take a sample in the hyper space.

    This method is stateless, so the distribution of the samples is different
    from that of `tune` call. This function just gives a general idea of what
    a sample from the `space` looks like.

    Example:
        >>> import matchzoo as mz
        >>> space = mz.models.Naive.get_default_params().hyper_space
        >>> mz.hyper_spaces.sample(space)  # doctest: +ELLIPSIS
        {'optimizer': ...}

    """
    return hyperopt.pyll.stochastic.sample(space)
