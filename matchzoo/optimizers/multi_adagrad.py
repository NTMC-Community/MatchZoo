"""Adagrad optimizer wiht LR multipliers."""
import typing

import keras
import keras.backend as K
from keras.legacy import interfaces

import matchzoo
from matchzoo.optimizers import MultiOptimizer


class MultiAdagrad(MultiOptimizer):
    """Adagrad optimizer with LR multipliers.

    Adagrad is an optimizer with parameter-specific learning rates,
    which are adapted relative to how frequently a parameter gets
    updated during training. The more updates a parameter receives,
    the smaller the updates.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    :param lr: Float >= 0. Initial learning rate.
    :param epsilon: Float >= 0. If `None`, defaults to `K.epsilon()`.
    :param decay: Float >= 0. Learning rate decay over each update.

    :param nesterov: Bool. Whether to apply Nesterov momentum.
    :param multipliers: Dict. Different learning rate multiplier for
        different layers. For example, `multipliers={'dense_1':0.8,
        'conv_1/kernel':0.5}.

    Example:
        >>> import matchzoo as mz
        >>> multi_optimizer = mz.optimizer.MultiAdagrad(
        ...    multipliers={'dense_1':0.8, 'conv_1/kernel':0.5}
        ... )

    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic
           Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """

    def __init__(self, lr=0.01, epsilon=None, decay=0.,
                 multipliers=None, **kwargs):
        super().__init__()
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):

            # Get learning rate multiplier
            multiplier = self.get_multiplier(p)

            # Get new learning rate
            new_lr = lr
            if multiplier:
                new_lr = lr * multiplier

            new_a = a + K.square(g)  # update accumulator
            self.updates.append(K.update(a, new_a))
            new_p = p - new_lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'multipliers': self.multipliers}
        base_config = super(MultiAdagrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
