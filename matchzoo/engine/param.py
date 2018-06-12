import inspect
import typing
import numbers


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
        <hyperopt.pyll.base.Apply object at 0x...>
        >>> from hyperopt.pyll.stochastic import sample
        >>> samples = [sample(param.hyper_space) for _ in range(64)]
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

    A `_pre_assignment_hook` is initialized as a data type transformer if the
    value is set as a number to keep data type consistency of the parameter.
    This conversion supports python built-in numbers and `numpy` numbers.

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
            value=None,
            hyper_space=None,
            validator=None,
    ):
        self._name = name

        self._value = None
        self._hyper_space = None
        self._validator = None
        self._pre_assignment_hook = None

        self.validator = validator
        self.hyper_space = hyper_space
        self.value = value

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if self._pre_assignment_hook:
            new_value = self._pre_assignment_hook(new_value)
        self._validate(new_value)
        self._value = new_value
        if not self._pre_assignment_hook:
            self._infer_pre_assignment_hook()

    @property
    def hyper_space(self):
        return self._hyper_space

    @hyper_space.setter
    def hyper_space(self, new_space):
        if hasattr(new_space, 'is_hyperopt_api'):
            new_space = new_space(self.name)
        self._hyper_space = new_space

    @property
    def validator(self):
        return self._validator

    @validator.setter
    def validator(self, new_validator):
        if new_validator and not callable(new_validator):
            raise TypeError("Validator must be a callable or None.")
        self._validator = new_validator

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
        return self._value is not None
