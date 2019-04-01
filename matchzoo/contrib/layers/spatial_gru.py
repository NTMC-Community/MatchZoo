"""An implementation of Spatial GRU Layer."""
import typing

from keras import backend as K
from keras.engine import Layer
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers import activations
from keras.layers import initializers

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops


class SpatialGRU(Layer):
    """
    Spatial GRU layer.

    :param units: Number of SpatialGRU units.
    :param normalize: Whether to L2-normalize samples along the
        dot product axis before taking the dot product.
        If set to True, then the output of the dot product
        is the cosine proximity between the two samples.
    :param init_diag: Whether to initialize the diagonal elements
        of the matrix.
    :param activation: Activation function to use. Default:
        hyperbolic tangent (`tanh`). If you pass `None`, no
        activation is applied (ie. "linear" activation: `a(x) = x`).
    :param recurrent_activation: Activation function to use for
        the recurrent step. Default: sigmoid (`sigmoid`).
        If you pass `None`, no activation is applied (ie. "linear"
        activation: `a(x) = x`).
    :param use_bias: Boolean, whether the layer uses a bias vector.
    :param kernel_initializer: Initializer for the `kernel` weights
        matrix, used for the linear transformation of the inputs.
    :param recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the
        recurrent state.
    :param bias_initializer: Initializer for the bias vector.
    :param direction: Scanning direction. `lt` (i.e., left top)
        indicates the scanning from left top to right bottom, and
        `rb` (i.e., right bottom) indicates the scanning from
        right bottom to left top.
    :param kwargs: Standard layer keyword arguments.

    Examples:
        >>> import matchzoo as mz
        >>> layer = mz.contrib.layers.SpatialGRU(units=50,
        ...                                      direction='lt')
        >>> num_batch, channel, left_len, right_len = 5, 10, 3, 2
        >>> layer.build([num_batch, channel, left_len, right_len])

    """

    def __init__(self,
                 units: int = 50,
                 normalize: bool = False,
                 init_diag: bool = False,
                 activation: str = 'tanh',
                 recurrent_activation: str = 'sigmoid',
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform',
                 recurrent_initializer: str = 'orthogonal',
                 bias_initializer: str = 'zeros',
                 direction: str = 'lr',
                 **kwargs):
        """:class:`SpatialGRU` constructor."""
        super(SpatialGRU, self).__init__(**kwargs)
        self._units = units
        self._normalize = normalize
        self._init_diag = init_diag
        self._activation = activations.get(activation)
        self._recurrent_activation = activations.get(recurrent_activation)
        self._use_bias = use_bias

        self._kernel_initializer = initializers.get(kernel_initializer)
        self._recurrent_initializer = initializers.get(recurrent_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._direction = direction

    def build(self, input_shape: typing.Any):
        """
        Build the layer.

        :param input_shape: the shapes of the input tensors.
        """
        # input_shape: (batch_size, channel, text1_maxlen, text2_maxlen)
        self._batch_size = input_shape[0]
        self._channel = input_shape[1]
        self._input_dim = self._channel + 3 * self._units

        self._text1_maxlen = input_shape[2]
        self._text2_maxlen = input_shape[3]
        self._recurrent_step = self._text1_maxlen * self._text2_maxlen

        self._W = self.add_weight(name='W',
                                  shape=(self._input_dim, self._units * 7),
                                  initializer=self._kernel_initializer,
                                  trainable=True)

        self._U = self.add_weight(name='U',
                                  shape=(self._units * 3, self._units),
                                  initializer=self._recurrent_initializer,
                                  trainable=True)

        self._bias = self.add_weight(name='bias',
                                     shape=(self._units * 8,),
                                     initializer='zeros',
                                     trainable=True)

        # w_rl, w_rt, w_rd
        self._wr = self._W[:, :self._units * 3]
        # b_rl, b_rt, b_rd
        self._br = self._bias[:self._units * 3]
        # w_zi, w_zl, w_zt, w_zd
        self._wz = self._W[:, self._units * 3: self._units * 7]
        # b_zi, b_zl, b_zt, b_zd
        self._bz = self._bias[self._units * 3: self._units * 7]
        self._w_ij = self.add_weight(name='Wij',
                                     shape=(self._channel, self._units),
                                     initializer=self._recurrent_initializer,
                                     trainable=True)
        self._b_ij = self._bias[self._units * 7:]
        super(SpatialGRU, self).build(input_shape)

    def softmax_by_row(self, z: typing.Any) -> tuple:
        """Conduct softmax on each dimension across the four gates."""

        z_transform = Permute((2, 1))(Reshape((4, self._units))(z))
        for i in range(0, self._units):
            begin1 = [0, i, 0]
            size = [-1, 1, -1]
            if i == 0:
                z_s = K.tf.nn.softmax(K.tf.slice(z_transform, begin1, size))
            else:
                z_s = K.tf.concat(
                    [z_s, K.tf.nn.softmax(
                        K.tf.slice(z_transform, begin1, size))], 1)

        zi, zl, zt, zd = K.tf.unstack(z_s, axis=2)
        return zi, zl, zt, zd

    def calculate_recurrent_unit(self,
                                 inputs_ta: typing.Any,
                                 states: typing.Any,
                                 step: int,
                                 h: typing.Any,
                                 h0: typing.Any) -> tuple:
        """
        Calculate recurrent unit.

        :param inputs: input tensors.
        """
        i = K.tf.div(step, K.tf.constant(self._text2_maxlen))
        j = K.tf.mod(step, K.tf.constant(self._text2_maxlen))

        h_diag = states.read(i * (self._text2_maxlen + 1) + j)
        h_top = states.read(i * (self._text2_maxlen + 1) + j + 1)
        h_left = states.read((i + 1) * (self._text2_maxlen + 1) + j)

        s_ij = inputs_ta.read(step)
        q = K.tf.concat([K.tf.concat([h_top, h_left], 1),
                        K.tf.concat([h_diag, s_ij], 1)], 1)
        r = self._recurrent_activation(
            self._time_distributed_dense(self._wr, q, self._br))
        z = self._time_distributed_dense(self._wz, q, self._bz)
        zi, zl, zt, zd = self.softmax_by_row(z)

        hij_1 = self._time_distributed_dense(self._w_ij, s_ij, self._b_ij)
        hij_2 = K.dot(r * (K.tf.concat([h_left, h_top, h_diag], 1)), self._U)
        hij_ = self._activation(hij_1 + hij_2)
        hij = zl * h_left + zt * h_top + zd * h_diag + zi * hij_
        states = states.write(((i + 1) * (self._text2_maxlen + 1) + j + 1),
                              hij)
        hij.set_shape(h_top.get_shape())
        return inputs_ta, states, step + 1, hij, h0

    def call(self, inputs: list, **kwargs) -> typing.Any:
        """
        The computation logic of SpatialGRU.

        :param inputs: input tensors.
        """
        batch_size = K.tf.shape(inputs)[0]
        self._bounder_state_h0 = K.tf.zeros([batch_size, self._units])

        # input_x: (text1_maxlen, text2_maxlen, batch_size, channel)
        input_x = K.tf.transpose(inputs, [2, 3, 0, 1])
        if self._direction == 'rb':
            input_x = K.tf.reverse(input_x, [0, 1])
        elif self._direction != 'lt':
            raise ValueError(f"Invalid direction. "
                             f"`{self._direction}` received. "
                             f"Must be in `lt`, `rb`.")
        input_x = K.tf.reshape(input_x, [-1, self._channel])
        input_x = K.tf.split(
            axis=0,
            num_or_size_splits=self._text1_maxlen * self._text2_maxlen,
            value=input_x)
        inputs_ta = K.tf.TensorArray(
            dtype=K.tf.float32,
            size=self._text1_maxlen * self._text2_maxlen,
            name='input_ta')
        states_ta = K.tf.TensorArray(
            dtype=K.tf.float32,
            size=(self._text1_maxlen + 1) * (self._text2_maxlen + 1),
            name='state_ta', clear_after_read=False)

        for i in range(self._text2_maxlen + 1):
            states_ta = states_ta.write(i, self._bounder_state_h0)
        for i in range(self._text1_maxlen):
            states_ta = states_ta.write((i + 1) * (self._text2_maxlen + 1),
                                        self._bounder_state_h0)
        inputs_ta = inputs_ta.unstack(input_x)
        _, _, _, hij, _ = control_flow_ops.while_loop(
            cond=lambda _0, _1, i, _3, _4: i < self._recurrent_step,
            body=self.calculate_recurrent_unit,
            loop_vars=(
                inputs_ta,
                states_ta,
                K.tf.Variable(0, dtype=K.tf.int32),
                self._bounder_state_h0,
                self._bounder_state_h0),
            parallel_iterations=1,
            swap_memory=True
        )
        return hij

    def compute_output_shape(self, input_shape: typing.Any) -> tuple:
        """
        Calculate the layer output shape.

        :param input_shape: the shapes of the input tensors.
        """
        output_shape = [input_shape[0], self._units]
        return tuple(output_shape)

    def get_config(self) -> dict:
        """Get the config dict of SpatialGRU."""
        config = {
            'units': self._units,
            'normalize': self._normalize,
            'init_diag': self._init_diag,
            'activation': activations.serialize(self._activation),
            'recurrent_activation':
                activations.serialize(self._recurrent_activation),
            'use_bias': self._use_bias,
            'kernel_initializer':
                initializers.serialize(self._kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self._recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self._bias_initializer),
            'direction': self._direction
        }
        base_config = super(SpatialGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def _time_distributed_dense(cls, w, x, b):
        if K.backend() == 'tensorflow':
            x = K.dot(x, w)
            x = K.bias_add(x, b)
        else:
            raise Exception("time_distributed_dense doesn't "
                            "backend tensorflow")
        return x
