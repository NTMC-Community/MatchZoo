from .multi_optimizer import MultiOptimizer
from .multi_sgd import MultiSGD
from .multi_rmsprop import MultiRMSprop
from .multi_adagrad import MultiAdagrad
from .multi_adadelta import MultiAdadelta
from .multi_adam import MultiAdam
from .multi_adamax import MultiAdamax
from .multi_nadam import MultiNadam


def list_available() -> list:
    from matchzoo.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(MultiOptimizer)
