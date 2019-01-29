"""Parameter class."""

import inspect
import numbers
import typing

import hyperopt.pyll

from matchzoo.engine import hyper_spaces

# Both hyperopt native spaces and matchzoo proxies are valid spaces.
SpaceType = typing.Union[hyperopt.pyll.Apply, hyper_spaces.HyperoptProxy]


class Param(object):
    """
    Parameter class.

    Basic usages with a name and  value:

        >>> param = Param('my_param', 10)
        >>> param.name
        'my_param'
        >>> param.value
        10

    Use with a validator to make sure the parameter always keeps a valid
    value.

        >>> param = Param(
        ...     name='my_param',
        ...     value=5,
        ...     validator=lambda x: 0 < x < 20
        ... )
        >>> param.validator  # doctest: +ELLIPSIS
        <function <lambda> at 0x...>
        >>> param.value
        5
        >>> param.value = 10
        >>> param.value
        10
        >>> param.value = -1
        Traceback (most recent call last):
            ...
        ValueError: Validator not satifised.
        The validator's definition is as follows:
        validator=lambda x: 0 < x < 20

    Use with a hyper space. Setting up a hyper space for a parameter makes the
    parameter tunable in a :class:`matchzoo.engine.Tuner`.

        >>> from matchzoo.engine.hyper_spaces import quniform
        >>> param = Param(
        ...     name='positive_num',
        ...     value=1,
        ...     hyper_space=quniform(low=1, high=5)
        ... )
        >>> param.hyper_space  # doctest: +ELLIPSIS
        <matchzoo.engine.hyper_spaces.quniform object at ...>
        >>> from hyperopt.pyll.stochastic import sample
        >>> hyperopt_space = param.hyper_space.convert(param.name)
        >>> samples = [sample(hyperopt_space) for _ in range(64)]
        >>> set(samples) == {1, 2, 3, 4, 5}
        True

    The boolean value of a :class:`Param` instance is only `True`
    when the value is not `None`. This is because some default falsy values
    like zero or an empty list are valid parameter values. In other words,
    the boolean value means to be "if the parameter value is filled".

        >>> param = Param('dropout')
        >>> if param:
        ...     print('OK')
        >>> param = Param('dropout', 0)
        >>> if param:
        ...     print('OK')
        OK

    A `_pre_assignment_hook` is initialized as a data type convertor if the
    value is set as a number to keep data type consistency of the parameter.
    This conversion supports python built-in numbers, `numpy` numbers, and
    any number that inherits :class:`numbers.Number`.

        >>> param = Param('float_param', 0.5)
        >>> param.value = 10
        >>> param.value
        10.0
        >>> type(param.value)
        <class 'float'>

    """

    def __init__(
        self,
        name: str,
        value: typing.Any = None,
        hyper_space: typing.Optional[SpaceType] = None,
        validator: typing.Optional[
            typing.Callable[[typing.Any], bool]] = None,
        desc: typing.Optional[str] = None,
    ):
        """
        Parameter constructor.

        :param name: Name of the parameter.
        :param value: Value of the parameter, `None` by default, which means
            "this parameter is not filled yet."
        :param hyper_space: Hyper space of the parameter, `None` by default.
            If set, then a :class:`matchzoo.engine.ParamTable` that has this
            parameter will include this `hyper_space` as a part of the
            parameter table's search space.
        :param validator: Validator of the parameter, `None` by default. If
            validation is needed, pass a callable that, given a value, returns
            a `bool`. The definition of the validator is retrieved when the
            validation fails, so either use a function or a `lambda` that
            occupies its own line for better readability.
        """
        self._name = name
        self._desc = desc

        self._value = None
        self._hyper_space = None
        self._validator = None
        self._pre_assignment_hook = None

        self.validator = validator
        self.hyper_space = hyper_space

        if value is not None:  # bypass checking if no default
            self.value = value

    @property
    def name(self) -> str:
        """:return: Name of the parameter."""
        return self._name

    @property
    def value(self) -> typing.Any:
        """:return: Value of the parameter."""
        return self._value

    @value.setter
    def value(self, new_value: typing.Any):
        """
        Set the value of parameter to `new_value`.

        Notice that this setter validates `new_value` before assignment. As
        a result, if the validaiton fails, the value of the parameter is not
        changed.

        :param new_value: New value of the parameter to set.
        """
        if self._pre_assignment_hook:
            new_value = self._pre_assignment_hook(new_value)
        self._validate(new_value)
        self._value = new_value
        if not self._pre_assignment_hook:
            self._infer_pre_assignment_hook()

    @property
    def hyper_space(self) -> SpaceType:
        """:return: Hyper space of the parameter."""
        return self._hyper_space

    @hyper_space.setter
    def hyper_space(self, new_space: SpaceType):
        """:param new_space: New space of the parameter to set."""
        self._hyper_space = new_space

    @property
    def validator(self) -> typing.Callable[[typing.Any], bool]:
        """:return: Validator of the parameter."""
        return self._validator

    @validator.setter
    def validator(self, new_validator: typing.Callable[[typing.Any], bool]):
        """:param new_validator: New space of the parameter to set."""
        if new_validator and not callable(new_validator):
            raise TypeError("Validator must be a callable or None.")
        self._validator = new_validator

    @property
    def desc(self) -> str:
        """:return: Parameter description."""
        return self._desc

    @desc.setter
    def desc(self, value: str):
        """:param value: New description of the parameter."""
        self._desc = value

    def _infer_pre_assignment_hook(self):
        if isinstance(self._value, numbers.Number):
            self._pre_assignment_hook = lambda x: type(self._value)(x)

    def _validate(self, value):
        if self._validator:
            valid = self._validator(value)
            if not valid:
                error_msg = "Validator not satifised.\n"
                error_msg += "The validator's definition is as follows:\n"
                error_msg += inspect.getsource(self._validator).strip()
                raise ValueError(error_msg)

    def __bool__(self):
        """:return: `False` when the value is `None`, `True` otherwise."""
        return self._value is not None

    def set_default(self, val, verbose=1):
        """
        Set default value, has no effect if already has a value.

        :param val: Default value to set.
        :param verbose: Verbosity.
        """
        if self._value is None:
            self.value = val
            if verbose:
                print(f"Parameter \"{self._name}\" set to {val}.")

    def reset(self):
        """
        Set the parameter's value to `None`, which means "not set".

        This method bypasses validator.

        Example:
            >>> import matchzoo as mz
            >>> param = mz.Param(
            ...     name='str', validator=lambda x: isinstance(x, str))
            >>> param.value = 'hello'
            >>> param.value = None
            Traceback (most recent call last):
                ...
            ValueError: Validator not satifised.
            The validator's definition is as follows:
            name='str', validator=lambda x: isinstance(x, str))
            >>> param.reset()
            >>> param.value is None
            True

        """
        self._value = None
