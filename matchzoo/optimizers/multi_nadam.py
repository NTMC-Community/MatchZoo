"""Nadam optimizer with LR multipliers."""
import typing

import keras
import keras.backend as K
from keras.legacy import interfaces

import matchzoo
from matchzoo.optimizers import MultiOptimizer


class MultiNadam(MultiOptimizer):
    """Nesterov Adam optimizer with LR multipliers.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.

    :param lr: Float >= 0. Learning rate.
    :param beta_1/beta_2: Floats, 0 < beta < 1. Generally close to 1.
    :param epsilon: Float >= 0. Fuzz factor. If `None`, defaults to
        `K.epsilon()`.
    :param multipliers: Dict. Different learning rate multiplier for
        different layers. For example, `multipliers={'dense_1':0.8,
        'conv_1/kernel':0.5}.

    Example:
        >>> import matchzoo as mz
        >>> multi_optimizer = mz.optimizers.MultiNadam(
        ...    multipliers={'dense_1':0.8, 'conv_1/kernel':0.5}
        ... )

    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning]
          (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999,
                 epsilon=None, schedule_decay=0.004,
                 multipliers=None, **kwargs):
        super().__init__()
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay
        self.multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * (
            momentum_cache_t_1)
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):

            # Get learning rate multiplier
            multiplier = self.get_multiplier(p)

            # Get new learning rate
            new_lr = self.lr
            if multiplier:
                new_lr = self.lr * multiplier

            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t_1 * m_t_prime)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            p_t = p - new_lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
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
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay,
                  'multipliers': self.multipliers}
        base_config = super(MultiNadam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
