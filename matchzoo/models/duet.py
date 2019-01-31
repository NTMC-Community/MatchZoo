"""DUET Model."""

import keras
import keras.backend as K
import tensorflow as tf

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class DUET(BaseModel):
    """
    DUET Model.

    Examples:
        >>> model = DUET()
        >>> model.params['embedding_input_dim'] = 1000
        >>> model.params['embedding_output_dim'] = 300
        >>> model.params['lm_filters'] = 32
        >>> model.params['lm_hidden_sizes'] = [64, 32]
        >>> model.params['dropout_rate'] = 0.5
        >>> model.params['dm_filters'] = 32
        >>> model.params['dm_kernel_size'] = 3
        >>> model.params['dm_d_mpool'] = 4
        >>> model.params['dm_hidden_sizes'] = [64, 32]
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls):
        """Get default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(name='lm_filters', value=32,
                         desc="Filter size of 1D convolution layer in "
                              "the local model."))
        params.add(Param(name='lm_hidden_sizes', value=[32],
                         desc="A list of hidden size of the MLP layer "
                              "in the local model."))
        params.add(Param(name='dm_filters', value=32,
                         desc="Filter size of 1D convolution layer in "
                              "the distributed model."))
        params.add(Param(name='dm_kernel_size', value=3,
                         desc="Kernel size of 1D convolution layer in "
                              "the distributed model."))
        params.add(Param(name='dm_q_hidden_size', value=32,
                         desc="Hidden size of the MLP layer for the "
                              "left text in the distributed model."))
        params.add(Param(name='dm_d_mpool', value=3,
                         desc="Max pooling size for the right text in "
                              "the distributed model."))
        params.add(Param(name='dm_hidden_sizes', value=[32],
                         desc="A list of hidden size of the MLP layer "
                              "in the distributed model."))
        params.add(Param(name='padding', value='same',
                         desc="The padding mode in the convolution "
                              "layer. It should be one of `same`, "
                              "`valid`, ""and `causal`."))
        params.add(Param(name='activation_func', value='relu',
                         desc="Activation function in the convolution"
                              " layer."))
        params.add(Param(
            name='dropout_rate', value=0.5,
            hyper_space=hyper_spaces.quniform(low=0.0, high=0.8,
                                              q=0.02),
            desc="The dropout rate."))
        return params

    def build(self):
        """Build model."""
        query, doc = self._make_inputs()

        embedding = self._make_embedding_layer()
        q_embed = embedding(query)
        d_embed = embedding(doc)

        lm_xor = keras.layers.Lambda(self._xor_match)([query, doc])
        lm_conv = keras.layers.Conv1D(
            self._params['lm_filters'],
            self._params['input_shapes'][1][0],
            padding=self._params['padding'],
            activation=self._params['activation_func']
        )(lm_xor)

        lm_conv = keras.layers.Dropout(self._params['dropout_rate'])(
            lm_conv)
        lm_feat = keras.layers.Reshape((-1,))(lm_conv)
        for hidden_size in self._params['lm_hidden_sizes']:
            lm_feat = keras.layers.Dense(
                hidden_size,
                activation=self._params['activation_func']
            )(lm_feat)
        lm_drop = keras.layers.Dropout(self._params['dropout_rate'])(
            lm_feat)
        lm_score = keras.layers.Dense(1)(lm_drop)

        dm_q_conv = keras.layers.Conv1D(
            self._params['dm_filters'],
            self._params['dm_kernel_size'],
            padding=self._params['padding'],
            activation=self._params['activation_func']
        )(q_embed)
        dm_q_conv = keras.layers.Dropout(self._params['dropout_rate'])(
            dm_q_conv)
        dm_q_mp = keras.layers.MaxPooling1D(
            pool_size=self._params['input_shapes'][0][0])(dm_q_conv)
        dm_q_rep = keras.layers.Reshape((-1,))(dm_q_mp)
        dm_q_rep = keras.layers.Dense(self._params['dm_q_hidden_size'])(
            dm_q_rep)
        dm_q_rep = keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(
            dm_q_rep)

        dm_d_conv1 = keras.layers.Conv1D(
            self._params['dm_filters'],
            self._params['dm_kernel_size'],
            padding=self._params['padding'],
            activation=self._params['activation_func']
        )(d_embed)
        dm_d_conv1 = keras.layers.Dropout(self._params['dropout_rate'])(
            dm_d_conv1)
        dm_d_mp = keras.layers.MaxPooling1D(
            pool_size=self._params['dm_d_mpool'])(dm_d_conv1)
        dm_d_conv2 = keras.layers.Conv1D(
            self._params['dm_filters'], 1,
            padding=self._params['padding'],
            activation=self._params['activation_func']
        )(dm_d_mp)
        dm_d_conv2 = keras.layers.Dropout(self._params['dropout_rate'])(
            dm_d_conv2)

        h_dot = keras.layers.Lambda(self._hadamard_dot)([dm_q_rep, dm_d_conv2])
        dm_feat = keras.layers.Reshape((-1,))(h_dot)
        for hidden_size in self._params['dm_hidden_sizes']:
            dm_feat = keras.layers.Dense(hidden_size)(dm_feat)
        dm_feat_drop = keras.layers.Dropout(self._params['dropout_rate'])(
            dm_feat)
        dm_score = keras.layers.Dense(1)(dm_feat_drop)

        add = keras.layers.Add()([lm_score, dm_score])
        out = self._make_output_layer()(add)
        self._backend = keras.Model(inputs=[query, doc], outputs=out)

    @classmethod
    def _xor_match(cls, x):
        t1 = x[0]
        t2 = x[1]
        t1_shape = t1.get_shape()
        t2_shape = t2.get_shape()
        t1_expand = K.tf.stack([t1] * t2_shape[1], 2)
        t2_expand = K.tf.stack([t2] * t1_shape[1], 1)
        out_bool = K.tf.equal(t1_expand, t2_expand)
        out = K.tf.cast(out_bool, K.tf.float32)
        return out

    @classmethod
    def _hadamard_dot(cls, x):
        x1 = x[0]
        x2 = x[1]
        out = x1 * x2
        return out
