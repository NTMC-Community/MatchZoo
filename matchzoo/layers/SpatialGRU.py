from __future__ import absolute_import
from keras import backend as K
from keras.engine import Layer
from keras.layers import Reshape, Permute
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import tensorflow as tf
from keras.layers import activations
from keras.layers import initializers
from keras.layers import regularizers
from keras.layers import constraints


def _time_distributed_dense(w, x, b):
    if K.backend() == 'tensorflow':
        x = K.dot(x, w)
        x = K.bias_add(x, b)
    else:
        print("time_distributed_dense doesn't backend tensorflow")
    return x


class SpatialGRU(Layer):
    # @interfaces.legacy_recurrent_support
    def __init__(self,
                 units=50,
                 normalize=False,
                 init_diag=False,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 **kwargs):

        super(SpatialGRU, self).__init__(**kwargs)
        self.units = units
        self.normalize = normalize
        self.init_diag = init_diag
        self.supports_masking = True
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        self.batch_size = input_shape[0]  # if self.stateful else None
        self.channel = input_shape[1]
        self.input_dim = self.channel + 3 * self.units

        self.text1_maxlen = input_shape[2]
        self.text2_maxlen = input_shape[3]
        self.recurrent_step = self.text1_maxlen * self.text2_maxlen

        self.W = self.add_weight(name='W',
                                 shape=(self.input_dim, self.units * 7),
                                 initializer=self.kernel_initializer)

        self.U = self.add_weight(name='U',
                                 shape=(self.units * 3, self.units),
                                 initializer=self.recurrent_initializer)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.units * 8,),
                                    initializer='zeros',
                                    trainable=True)

        self.wr = self.W[:, :self.units * 3]
        self.br = self.bias[:self.units * 3]
        self.wz = self.W[:, self.units * 3: self.units * 7]
        self.bz = self.bias[self.units * 3: self.units * 7]
        self.w_ij = self.add_weight(name='Wij',
                                    shape=(self.channel, self.units),
                                    initializer=self.recurrent_initializer)
        self.b_ij = self.bias[self.units * 7:]

    def softmax_by_row(self, z):
        z_transform = Permute((2, 1))(Reshape((4, self.units))(z))
        for i in range(0, self.units):
            begin1 = [0, i, 0]
            size = [-1, 1, -1]
            if i == 0:
                z_s = tf.nn.softmax(tf.slice(z_transform, begin1, size))
            else:
                z_s = tf.concat([z_s, tf.nn.softmax(tf.slice(z_transform, begin1, size))], 1)

        print ('calculate---z_s---shape', z_s)
        zi, zl, zt, zd = tf.unstack(z_s, axis=2)
        return zi, zl, zt, zd

    def calculate_recurrent_unit(self, inputs_ta, states, step, h, h0):
        i = tf.div(step, tf.constant(self.text2_maxlen))
        j = tf.mod(step, tf.constant(self.text2_maxlen))

        h_diag = states.read(i * (self.text2_maxlen + 1) + j)
        h_top = states.read(i * (self.text2_maxlen + 1) + j + 1)
        h_left = states.read((i + 1) * (self.text2_maxlen + 1) + j)

        s_ij = inputs_ta.read(step)
        q = tf.concat([tf.concat([h_top, h_left], 1), tf.concat([h_diag, s_ij], 1)], 1)
        r = self.recurrent_activation(_time_distributed_dense(self.wr, q, self.br))
        z = _time_distributed_dense(self.wz, q, self.bz)
        zi, zl, zt, zd = self.softmax_by_row(z)

        hij_ = self.activation(_time_distributed_dense(self.w_ij, s_ij, self.b_ij) +
                               K.dot(r * (tf.concat([h_left, h_top, h_diag], 1)), self.U))
        hij = zl * h_left + zt * h_top + zd * h_diag + zi * hij_
        states = states.write(((i + 1) * (self.text2_maxlen + 1) + j + 1), hij)
        hij.set_shape(h_top.get_shape())
        return inputs_ta, states, step + 1, hij, h0

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        self.bounder_state_h0 = tf.zeros([batch_size, self.units])

        input_x = tf.transpose(inputs, [2, 3, 0, 1])
        input_x = tf.reshape(input_x, [-1, self.channel])
        input_x = tf.split(axis=0, num_or_size_splits=self.text1_maxlen * self.text2_maxlen, value=input_x)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=self.text1_maxlen * self.text2_maxlen, name='input_ta')
        states_ta = tf.TensorArray(dtype=tf.float32, size=(self.text1_maxlen + 1) * (self.text2_maxlen + 1),
                                   name='state_ta', clear_after_read=False)

        for i in range(self.text2_maxlen + 1):
            states_ta = states_ta.write(i, self.bounder_state_h0)
        for i in range(self.text1_maxlen):
            states_ta = states_ta.write((i + 1) * (self.text2_maxlen + 1), self.bounder_state_h0)
        inputs_ta = inputs_ta.unstack(input_x)
        _, _, _, hij, _ = control_flow_ops.while_loop(
            cond=lambda _0, _1, i, _3, _4: i < self.recurrent_step,
            body=self.calculate_recurrent_unit,
            loop_vars=(
                inputs_ta, states_ta, tf.Variable(0, dtype=tf.int32), self.bounder_state_h0, self.bounder_state_h0),
            parallel_iterations=1,
            swap_memory=True
        )
        return hij

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], self.units]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'channel': self.channel,
            'normalize': self.normalize,
            'init_diag': self.init_diag,
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer)
        }
        base_config = super(SpatialGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
