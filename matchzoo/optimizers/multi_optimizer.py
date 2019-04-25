"""Optimizer with LR multipliers."""
import typing

import keras
import keras.backend as K
from keras.optimizers import Optimizer
from keras.legacy import interfaces


class MultiOptimizer(Optimizer):
    """Abstract optimizer with LR multipliers base class.

    Note: this is the parent class of all multi optimizers, not an actual
    optimizer that can be used for training models.
    """

    def __init__(self, **kwargs):
        """Initialization."""
        super().__init__()

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        """Update params."""
        raise NotImplementedError

    def get_multiplier(self, param) -> float:
        """Get multiplier."""
        if self.multipliers is not None:
            for k in self.multipliers.keys():
                if k in param.name:
                    return self.multipliers[k]
            return None
        return None
