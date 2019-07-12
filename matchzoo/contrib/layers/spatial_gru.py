"""An implementation of Spatial GRU Layer."""
import typing

from keras import backend as K
from keras.engine import Layer
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers import activations
from keras.layers import initializers


class SpatialGRU(Layer):
    """
    Spatial GRU layer.

    :param units: Number of SpatialGRU units.
    :param activation: Activation function to use. Default:
        hyperbolic tangent (`tanh`). If you pass `None`, no
        activation is applied (ie. "linear" activation: `a(x) = x`).
    :param recurrent_activation: Activation function to use for
        the recurrent step. Default: sigmoid (`sigmoid`).
        If you pass `None`, no activation is applied (ie. "linear"
        activation: `a(x) = x`).
    :param kernel_initializer: Initializer for the `kernel` weights
        matrix, used for the linear transformation of the inputs.
    :param recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the
        recurrent state.
    :param direction: Scanning direction. `lt` (i.e., left top)
        indicates the scanning from left top to right bottom, and
        `rb` (i.e., right bottom) indicates the scanning from
        right bottom to left top.
    :param kwargs: Standard layer keyword arguments.

    Examples:
        >>> import matchzoo as mz
        >>> layer = mz.contrib.layers.SpatialGRU(units=10,
        ...                                      direction='lt')
        >>> num_batch, channel, left_len, right_len = 5, 5, 3, 2
        >>> layer.build([num_batch, channel, left_len, right_len])

    """

    def __init__(
        self,
        units: int = 10,
        activation: str = 'tanh',
        recurrent_activation: str = 'sigmoid',
        kernel_initializer: str = 'glorot_uniform',
        recurrent_initializer: str = 'orthogonal',
        direction: str = 'lt',
        **kwargs
    ):
        """:class:`SpatialGRU` constructor."""
        super().__init__(**kwargs)
        self._units = units
        self._activation = activations.get(activation)
        self._recurrent_activation = activations.get(recurrent_activation)

        self._kernel_initializer = initializers.get(kernel_initializer)
        self._recurrent_initializer = initializers.get(recurrent_initializer)
        self._direction = direction

    def build(self, input_shape: typing.Any):
        """
        Build the layer.

        :param input_shape: the shapes of the input tensors.
        """
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   C = number of channels
        #   U = number of units

        # input_shape = [B, C, L, R]
        self._batch_size = input_shape[0]
        self._channel = input_shape[1]
        self._input_dim = self._channel + 3 * self._units

        self._text1_maxlen = input_shape[2]
        self._text2_maxlen = input_shape[3]
        self._recurrent_step = self._text1_maxlen * self._text2_maxlen
        # W = [3*U+C, 7*U]
        self._W = self.add_weight(
            name='W',
            shape=(self._input_dim, self._units * 7),
            initializer=self._kernel_initializer,
            trainable=True
        )
        # U = [3*U, U]
        self._U = self.add_weight(
            name='U',
            shape=(self._units * 3, self._units),
            initializer=self._recurrent_initializer,
            trainable=True
        )
        # bias = [8*U,]
        self._bias = self.add_weight(
            name='bias',
            shape=(self._units * 8,),
            initializer='zeros',
            trainable=True
        )

        # w_rl, w_rt, w_rd = [B, 3*U]
        self._wr = self._W[:, :self._units * 3]
        # b_rl, b_rt, b_rd = [B, 3*U]
        self._br = self._bias[:self._units * 3]
        # w_zi, w_zl, w_zt, w_zd = [B, 4*U]
        self._wz = self._W[:, self._units * 3: self._units * 7]
        # b_zi, b_zl, b_zt, b_zd = [B, 4*U]
        self._bz = self._bias[self._units * 3: self._units * 7]
        # w_ij = [C, U]
        self._w_ij = self.add_weight(
            name='W_ij',
            shape=(self._channel, self._units),
            initializer=self._recurrent_initializer,
            trainable=True
        )
        # b_ij = [7*U]
        self._b_ij = self._bias[self._units * 7:]
        super(SpatialGRU, self).build(input_shape)

    def softmax_by_row(self, z: typing.Any) -> tuple:
        """Conduct softmax on each dimension across the four gates."""

        # z_transform: [B, U, 4]
        z_transform = Permute((2, 1))(Reshape((4, self._units))(z))
        size = [-1, 1, -1]
        # Perform softmax on each slice
        for i in range(0, self._units):
            begin = [0, i, 0]
            # z_slice: [B, 1, 4]
            z_slice = K.tf.slice(z_transform, begin, size)
            if i == 0:
                z_s = K.tf.nn.softmax(z_slice)
            else:
                z_s = K.tf.concat([z_s, K.tf.nn.softmax(z_slice)], 1)
        # zi, zl, zt, zd: [B, U]
        zi, zl, zt, zd = K.tf.unstack(z_s, axis=2)
        return zi, zl, zt, zd

    def calculate_recurrent_unit(
        self,
        inputs: typing.Any,
        states: typing.Any,
        step: int,
        h: typing.Any,
    ) -> tuple:
        """
        Calculate recurrent unit.

        :param inputs: A TensorArray which contains interaction
            between left text and right text.
        :param states: A TensorArray which stores the hidden state
            of every step.
        :param step: Recurrent step.
        :param h: Hidden state from last operation.
        """
        # Get index i, j
        i = K.tf.floordiv(step, K.tf.constant(self._text2_maxlen))
        j = K.tf.mod(step, K.tf.constant(self._text2_maxlen))

        # Get hidden state h_diag, h_top, h_left
        # h_diag, h_top, h_left = [B, U]
        h_diag = states.read(i * (self._text2_maxlen + 1) + j)
        h_top = states.read(i * (self._text2_maxlen + 1) + j + 1)
        h_left = states.read((i + 1) * (self._text2_maxlen + 1) + j)

        # Get interaction between word i, j: s_ij
        # s_ij = [B, C]
        s_ij = inputs.read(step)

        # Concatenate h_top, h_left, h_diag, s_ij
        # q = [B, 3*U+C]
        q = K.tf.concat([K.tf.concat([h_top, h_left], 1),
                        K.tf.concat([h_diag, s_ij], 1)], 1)

        # Calculate reset gate
        # r = [B, 3*U]
        r = self._recurrent_activation(
            self._time_distributed_dense(self._wr, q, self._br))

        # Calculate updating gate
        # z: [B, 4*U]
        z = self._time_distributed_dense(self._wz, q, self._bz)

        # Perform softmax
        # zi, zl, zt, zd: [B, U]
        zi, zl, zt, zd = self.softmax_by_row(z)

        # Get h_ij_
        # h_ij_ = [B, U]
        h_ij_l = self._time_distributed_dense(self._w_ij, s_ij, self._b_ij)
        h_ij_r = K.dot(r * (K.tf.concat([h_left, h_top, h_diag], 1)), self._U)
        h_ij_ = self._activation(h_ij_l + h_ij_r)

        # Calculate h_ij
        # h_ij = [B, U]
        h_ij = zl * h_left + zt * h_top + zd * h_diag + zi * h_ij_

        # Write h_ij to states
        states = states.write(((i + 1) * (self._text2_maxlen + 1) + j + 1),
                              h_ij)
        h_ij.set_shape(h_top.get_shape())

        return inputs, states, step + 1, h_ij

    def call(self, inputs: list, **kwargs) -> typing.Any:
        """
        The computation logic of SpatialGRU.

        :param inputs: input tensors.
        """
        batch_size = K.tf.shape(inputs)[0]
        # h0 = [B, U]
        self._bounder_state_h0 = K.tf.zeros([batch_size, self._units])

        # input_x = [L, R, B, C]
        input_x = K.tf.transpose(inputs, [2, 3, 0, 1])
        if self._direction == 'rb':
            # input_x: [R, L, B, C]
            input_x = K.tf.reverse(input_x, [0, 1])
        elif self._direction != 'lt':
            raise ValueError(f"Invalid direction. "
                             f"`{self._direction}` received. "
                             f"Must be in `lt`, `rb`.")
        # input_x = [L*R*B, C]
        input_x = K.tf.reshape(input_x, [-1, self._channel])
        # input_x = L*R * [B, C]
        input_x = K.tf.split(
            axis=0,
            num_or_size_splits=self._text1_maxlen * self._text2_maxlen,
            value=input_x
        )

        # inputs = L*R * [B, C]
        inputs = K.tf.TensorArray(
            dtype=K.tf.float32,
            size=self._text1_maxlen * self._text2_maxlen,
            name='inputs'
        )
        inputs = inputs.unstack(input_x)

        # states = (L+1)*(R+1) * [B, U]
        states = K.tf.TensorArray(
            dtype=K.tf.float32,
            size=(self._text1_maxlen + 1) * (self._text2_maxlen + 1),
            name='states',
            clear_after_read=False
        )
        # Initialize states
        for i in range(self._text2_maxlen + 1):
            states = states.write(i, self._bounder_state_h0)
        for i in range(1, self._text1_maxlen + 1):
            states = states.write(i * (self._text2_maxlen + 1),
                                  self._bounder_state_h0)

        # Calculate h_ij
        # h_ij = [B, U]
        _, _, _, h_ij = K.tf.while_loop(
            cond=lambda _0, _1, i, _3: K.tf.less(i, self._recurrent_step),
            body=self.calculate_recurrent_unit,
            loop_vars=(
                inputs,
                states,
                K.tf.constant(0, dtype=K.tf.int32),
                self._bounder_state_h0
            ),
            parallel_iterations=1,
            swap_memory=True
        )
        return h_ij

    def compute_output_shape(self, input_shape: typing.Any) -> tuple:
        """
        Calculate the layer output shape.

        :param input_shape: the shapes of the input tensors.
        """
        output_shape = [input_shape[0], self._units]
        return tuple(output_shape)

    @classmethod
    def _time_distributed_dense(cls, w, x, b):
        x = K.dot(x, w)
        x = K.bias_add(x, b)
        return x
