"""An implementation of Decaying Dropout Layer."""

import tensorflow as tf
from keras import backend as K
from keras.engine import Layer

class DecayingDropoutLayer(Layer):
    """
    Layer that processes dropout with exponential decayed keep rate during
    training.

    :param initial_keep_rate: the initial keep rate of decaying dropout.
    :param decay_interval: the decay interval of decaying dropout.
    :param decay_rate: the decay rate of decaying dropout.
    :param noise_shape: a 1D integer tensor representing the shape of the
        binary dropout mask that will be multiplied with the input.
    :param seed: a python integer to use as random seed.
    :param kwargs: standard layer keyword arguments.

    Examples:
        >>> import matchzoo as mz
        >>> layer = mz.contrib.layers.DecayingDropoutLayer(
        ...     initial_keep_rate=1.0,
        ...     decay_interval=10000,
        ...     decay_rate=0.977,
        ... )
        >>> num_batch, num_dim =5, 10
        >>> layer.build([num_batch, num_dim])
    """

    def __init__(self,
                 initial_keep_rate: float = 1.0,
                 decay_interval: int = 10000,
                 decay_rate: float = 0.977,
                 noise_shape=None,
                 seed=None,
                 **kwargs):
        """:class: 'DecayingDropoutLayer' constructor."""
        super(DecayingDropoutLayer, self).__init__(**kwargs)
        self._iterations = None
        self._initial_keep_rate = initial_keep_rate
        self._decay_interval = decay_interval
        self._decay_rate = min(1.0, max(0.0, decay_rate))
        self._noise_shape = noise_shape
        self._seed = seed

    def _get_noise_shape(self, inputs):
        if self._noise_shape is None:
            return self._noise_shape

        symbolic_shape = tf.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self._noise_shape)]
        return tuple(noise_shape)

    def build(self, input_shape):
        """
        Build the layer.

        :param input_shape: the shape of the input tensor,
            for DecayingDropoutLayer we need one input tensor.
        """

        self._iterations = self.add_weight(name='iterations',
                                           shape=(1,),
                                           dtype=K.floatx(),
                                           initializer='zeros',
                                           trainable=False)
        super(DecayingDropoutLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        The computation logic of DecayingDropoutLayer.

        :param inputs: an input tensor.
        """
        noise_shape = self._get_noise_shape(inputs)
        t = tf.cast(self._iterations, K.floatx()) + 1
        p = t / float(self._decay_interval)

        keep_rate = self._initial_keep_rate * tf.pow(self._decay_rate, p)

        def dropped_inputs():
            update_op = self._iterations.assign_add([1])
            with tf.control_dependencies([update_op]):
                return tf.nn.dropout(inputs, 1 - keep_rate[0], noise_shape,
                                 seed=self._seed)

        return K.in_train_phase(dropped_inputs, inputs, training=training)

    def get_config(self):
        """Get the config dict of DecayingDropoutLayer."""
        config = {'initial_keep_rate': self._initial_keep_rate,
                  'decay_interval': self._decay_interval,
                  'decay_rate': self._decay_rate,
                  'noise_shape': self._noise_shape,
                  'seed': self._seed}
        base_config = super(DecayingDropoutLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
