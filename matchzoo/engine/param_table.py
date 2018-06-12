"""Model parametrs."""

from matchzoo.engine import Param


class ParamTable(object):
    """


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
        self._params = {}

    def add(self, param):
        """

        :param param:

        """
        if not isinstance(param, Param):
            raise TypeError("Only accepts Param instance.")
        self._params[param.name] = param

    @property
    def hyper_space(self):
        return {
            param.name: param.hyper_space
            for param in self._params.values()
            if param.hyper_space is not None
        }

    def __getitem__(self, key):
        return self._params[key].value

    def __setitem__(self, key, value):
        self._params[key].value = value

    def __str__(self):
        return '\n'.join(param.name.ljust(30) + str(param.value)
                         for param in self._params.values())

    def __iter__(self):
        yield from self._params.values()
