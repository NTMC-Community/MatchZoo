"""Adadelta optimizer wiht LR multipliers."""
import typing

import keras
import keras.backend as K
from keras.legacy import interfaces

import matchzoo
from matchzoo.optimizers import MultiOptimizer


class MultiAdadelta(MultiOptimizer):
    """Adadelta optimizer with LR multipliers.

    Adadelta is a more robust extension of Adagrad
    that adapts learning rates based on a moving window of gradient updates,
    instead of accumulating all past gradients. This way, Adadelta continues
    learning even when many updates have been done. Compared to Adagrad, in the
    original version of Adadelta you don't have to set an initial learning
    rate. In this version, initial learning rate and decay factor can
    be set, as in most other Keras optimizers.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    :param lr: Float >= 0. Initial learning rate, defaults to 1.
        It is recommended to leave it at the default value.
    :param rho: Float >= 0. Adadelta decay factor, corresponding to fraction of
        gradient to keep at each time step.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Initial learning rate decay.
    :param multipliers: Dict. Different learning rate multiplier for
        different layers. For example, `multipliers={'dense_1':0.8,
        'conv_1/kernel':0.5}.

    Example:
        >>> import matchzoo as mz
        >>> multi_optimizer = mz.optimizers.MultiAdadelta(
        ...    multipliers={'dense_1':0.8, 'conv_1/kernel':0.5}
        ... )

    # References
        - [Adadelta - an adaptive learning rate method]
          (https://arxiv.org/abs/1212.5701)
    """

    def __init__(self, lr=1.0, rho=0.95, epsilon=None, decay=0.,
                 multipliers=None, **kwargs):
        super().__init__()
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.rho = rho
        self.epsilon = epsilon
        self.initial_decay = decay
        self.multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        delta_accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators + delta_accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a, d_a in zip(params, grads, accumulators,
                                delta_accumulators):

            # Get learning rate multiplier
            multiplier = self.get_multiplier(p)

            # Get new learning rate
            new_lr = lr
            if multiplier:
                new_lr = lr * multiplier

            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * K.sqrt(d_a + self.epsilon) / (
                K.sqrt(new_a + self.epsilon))
            new_p = p - new_lr * update

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
            self.updates.append(K.update(d_a, new_d_a))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': self.rho,
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'multipliers': self.multipliers}
        base_config = super(MultiAdadelta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
