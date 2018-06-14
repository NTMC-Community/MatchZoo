"""Parameters table class."""

import typing

from matchzoo.engine import Param


class ParamTable(object):
    """
    Parameter table class.

    Example:

        >>> params = ParamTable()
        >>> params.add(Param('ham', 'Parma Ham'))
        >>> params.add(Param('egg', 'Over Easy'))
        >>> params['ham']
        'Parma Ham'
        >>> params['egg']
        'Over Easy'
        >>> print(params)
        ham                           Parma Ham
        egg                           Over Easy

    """

    def __init__(self):
        """Parameter table constrctor."""
        self._params = {}

    def add(self, param: Param):
        """:param param: parameter to add."""
        if not isinstance(param, Param):
            raise TypeError("Only accepts Param instance.")
        self._params[param.name] = param

    @property
    def hyper_space(self) -> dict:
        """:return: Hyper space of the table, a valid `hyperopt` graph."""
        return {
            param.name: param.hyper_space
            for param in self._params.values()
            if param.hyper_space is not None
        }

    def __getitem__(self, key: str) -> typing.Any:
        """:return: A parameter instance in the table named `key`."""
        return self._params[key].value

    def __setitem__(self, key: str, value: typing.Any):
        """
        Set the value of the parameter named `key`.

        :param key: Name of the parameter.
        :param value: New value of the parameter to set.
        """
        self._params[key].value = value

    def __str__(self):
        """:return: Pretty formatted parameter table."""
        return '\n'.join(param.name.ljust(30) + str(param.value)
                         for param in self._params.values())

    def __iter__(self) -> typing.Iterator:
        """:return: A iterator that iterates over all parameter instances."""
        yield from self._params.values()
