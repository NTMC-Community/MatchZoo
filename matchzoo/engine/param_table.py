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
        >>> params.add(Param('egg', 'Sunny side Up'))
        Traceback (most recent call last):
            ...
        ValueError: Parameter named egg already exists.
        To re-assign parameter egg value, use `params["egg"] = value` instead.
    """

    def __init__(self):
        """Parameter table constrctor."""
        self._params = {}

    def add(self, param: Param):
        """:param param: parameter to add."""
        if not isinstance(param, Param):
            raise TypeError("Only accepts a Param instance.")
        if param.name in self._params:
            msg = f"Parameter named {param.name} already exists.\n" \
                  f"To re-assign parameter {param.name} value, " \
                  f"use `params[\"{param.name}\"] = value` instead."
            raise ValueError(msg)
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

    def completed(self) -> bool:
        """
        :return: `True` if all params are filled, `False` otherwise.

        Example:

            >>> import matchzoo
            >>> model = matchzoo.models.NaiveModel()
            >>> model.params.completed()
            False
            >>> model.guess_and_fill_missing_params()
            >>> model.params.completed()
            True

        """
        return all(param for param in self)
