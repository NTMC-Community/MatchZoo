"""Adamax optimizer with LR multipliers."""
import typing

import keras
import keras.backend as K
from keras.legacy import interfaces

import matchzoo
from matchzoo.optimizers import MultiOptimizer


class MultiAdamax(MultiOptimizer):
    """Adamax optimizer from Adam paper's Section 7 with LR multipliers.

    It is a variant of Adam based on the infinity norm.
    Default parameters follow those provided in the paper.

    :param lr: Float >= 0. Learning rate.
    :param beta_1/beta_2: Floats, 0 < beta < 1. Generally close to 1.
    :param epsilon: Float >= 0. Fuzz factor. If `None`, defaults to
        `K.epsilon()`.
    :param decay: float >= 0. Learning rate decay over each update.

    Example:
        >>> import matchzoo as mz
        >>> multi_optimizer = mz.optimizers.MultiAdamax(
        ...    multipliers={'dense_1':0.8, 'conv_1/kernel':0.5}
        ... )

    # References
        - [Adam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., multipliers=None,
                 **kwargs):
        super().__init__()
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr / (1. - K.pow(self.beta_1, t))

        shapes = [K.int_shape(p) for p in params]
        # zero init of 1st moment
        ms = [K.zeros(shape) for shape in shapes]
        # zero init of exponentially weighted infinity norm
        us = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + us

        for p, g, m, u in zip(params, grads, ms, us):

            # Get learning rate multiplier
            multiplier = self.get_multiplier(p)

            # Get new learning rate
            new_lr_t = lr_t
            if multiplier:
                new_lr_t = lr_t * multiplier

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            u_t = K.maximum(self.beta_2 * u, K.abs(g))
            p_t = p - new_lr_t * m_t / (u_t + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(u, u_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'multipliers': self.multipliers}
        base_config = super(MultiAdamax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
