"""RMSProp optimizer wiht LR multipliers."""
import typing

import keras
import keras.backend as K
from keras.legacy import interfaces

import matchzoo
from matchzoo.optimizers import MultiOptimizer


class MultiRMSprop(MultiOptimizer):
    """RMSProp optimizer with LR multipliers.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    This optimizer is usually a good choice for recurrent
    neural networks.

    :param lr: Float >= 0. Learning rate.
    :param rho: Float >= 0.
    :param epsilon: Float >= 0. Fuzz factor. If `None`, defaults to
        `K.epsilon()`.
    :param decay: Float >= 0. Learning rate decay over each update.
    :param multipliers: Dict. Different learning rate multiplier for
        different layers. For example, `multipliers={'dense_1':0.8,
        'conv_1/kernel':0.5}.

    Example:
        >>> import matchzoo as mz
        >>> multi_optimizer = mz.optimizers.MultiRMSprop(
        ...    multipliers={'dense_1':0.8, 'conv_1/kernel':0.5}
        ... )

    # References
        - [rmsprop: Divide the gradient by a running average of its
           recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/
           slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0.,
                 multipliers=None, **kwargs):
        super().__init__()
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.rho = K.variable(rho, name='rho')
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
        accumulators = [
            K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params
        ]
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

            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - new_lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'multipliers': self.multipliers}
        base_config = super(MultiRMSprop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
