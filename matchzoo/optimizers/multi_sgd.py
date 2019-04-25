"""Stochastic gradient descent optimizer with LR multipliers."""
import typing

import keras
import keras.backend as K
from keras.legacy import interfaces

import matchzoo
from matchzoo.optimizers import MultiOptimizer


class MultiSGD(MultiOptimizer):
    """Stochastic gradient descent optimizer with LR multiplier.

    Includes support for momentum,
    learning rate decay, Nesterov momentum.

    :param lr: Float >= 0. Learning rate.
    :param momentum: Float >= 0. Parameter that accelerates SGD
        in the relevant direction and dampens oscillations.
    :param decay: Float >= 0. Learning rate decay over each update.
    :param nesterov: Bool. Whether to apply Nesterov momentum.
    :param multipliers: Dict. Different learning rate multiplier for
        different layers. For example, `multipliers={'dense_1':0.8,
        'conv_1/kernel':0.5}.

    Example:
        >>> import matchzoo as mz
        >>> multi_optimizer = mz.optimizer.MultiSGD(
        ...    multipliers={'dense_1':0.8, 'conv_1/kernel':0.5}
        ... )
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, multipliers=None, **kwargs):
        super().__init__()
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):

            # Get learning rate multiplier
            multiplier = self.get_multiplier(p)

            # Get new learning rate
            new_lr = lr
            if multiplier:
                new_lr = lr * multiplier

            v = self.momentum * m - new_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - new_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'multipliers': self.multipliers}
        base_config = super(MultiSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
