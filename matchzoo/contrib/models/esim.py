"""ESIM model."""

import keras
import keras.backend as K
import tensorflow as tf

import matchzoo as mz
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable


class ESIM(BaseModel):
    """
    ESIM model.

    Examples:
        >>> model = ESIM()
        >>> task = classification_task = mz.tasks.Classification(num_classes=2)
        >>> model.params['task'] = task
        >>> model.params['input_shapes'] = [(20, ), (40, )]
        >>> model.params['lstm_dim'] = 300
        >>> model.params['mlp_num_units'] = 300
        >>> model.params['embedding_input_dim'] =  5000
        >>> model.params['embedding_output_dim'] =  10
        >>> model.params['embedding_trainable'] = False
        >>> model.params['mlp_num_layers'] = 0
        >>> model.params['mlp_num_fan_out'] = 300
        >>> model.params['mlp_activation_func'] = 'tanh'
        >>> model.params['mask_value'] = 0
        >>> model.params['dropout_rate'] = 0.5
        >>> model.params['optimizer'] = keras.optimizers.Adam(lr=4e-4)
        >>> model.guess_and_fill_missing_params()
        >>> model.build()
    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """Get default parameters."""
        params = super().get_default_params(with_embedding=True,
                                            with_multi_layer_perceptron=True)

        params.add(Param(
            name='dropout_rate',
            value=0.5,
            desc="The dropout rate for all fully-connected layer"
        ))

        params.add(Param(
            name='lstm_dim',
            value=8,
            desc="The dimension of LSTM layer."
        ))

        params.add(Param(
            name='mask_value',
            value=0,
            desc="The value would be regarded as pad"
        ))

        return params

    def _expand_dim(self, inp: tf.Tensor, axis: int) -> keras.layers.Layer:
        """
        Wrap keras.backend.expand_dims into a Lambda layer.

        :param inp: input tensor to expand the dimension
        :param axis: the axis of new dimension
        """
        return keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=axis))(inp)

    def _make_atten_mask_layer(self) -> keras.layers.Layer:
        """
        Make mask layer for attention weight matrix so that
        each word won't pay attention to <PAD> timestep.
        """
        return keras.layers.Lambda(
            lambda weight_mask: weight_mask[0] + (1.0 - weight_mask[1]) * -1e7,
            name="atten_mask")

    def _make_bilstm_layer(self, lstm_dim: int) -> keras.layers.Layer:
        """
        Bidirectional LSTM layer in ESIM.

        :param lstm_dim: int, dimension of LSTM layer
        :return: `keras.layers.Layer`.
        """
        return keras.layers.Bidirectional(
            layer=keras.layers.LSTM(lstm_dim, return_sequences=True),
            merge_mode='concat')

    def _max(self, texts: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Compute the max of each text according to their real length

        :param texts: np.array with shape [B, T, H]
        :param lengths: np.array with shape [B, T, ],
            where 1 means valid, 0 means pad
        """
        mask = self._expand_dim(mask, axis=2)
        new_texts = keras.layers.Multiply()([texts, mask])

        text_max = keras.layers.Lambda(
            lambda x: tf.reduce_max(x, axis=1),
        )(new_texts)

        return text_max

    def _avg(self, texts: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Compute the mean of each text according to their real length

        :param texts: np.array with shape [B, T, H]
        :param lengths: np.array with shape [B, T, ],
            where 1 means valid, 0 means pad
        """
        mask = self._expand_dim(mask, axis=2)
        new_texts = keras.layers.Multiply()([texts, mask])

        # timestep-wise division, exclude the PAD number when calc avg
        text_avg = keras.layers.Lambda(
            lambda text_mask:
                tf.reduce_sum(text_mask[0], axis=1) / tf.reduce_sum(text_mask[1], axis=1),
        )([new_texts, mask])

        return text_avg

    def build(self):
        """Build model."""
        # parameters
        lstm_dim = self._params['lstm_dim']
        dropout_rate = self._params['dropout_rate']

        # layers
        create_mask = keras.layers.Lambda(
            lambda x:
                tf.cast(tf.not_equal(x, self._params['mask_value']), K.floatx())
        )
        embedding = self._make_embedding_layer()
        lstm_compare = self._make_bilstm_layer(lstm_dim)
        lstm_compose = self._make_bilstm_layer(lstm_dim)
        dense_compare = keras.layers.Dense(units=lstm_dim,
                                           activation='relu',
                                           use_bias=True)
        dropout = keras.layers.Dropout(dropout_rate)

        # model
        a, b = self._make_inputs()      # [B, T_a], [B, T_b]
        a_mask = create_mask(a)         # [B, T_a]
        b_mask = create_mask(b)         # [B, T_b]

        # encoding
        a_emb = dropout(embedding(a))   # [B, T_a, E_dim]
        b_emb = dropout(embedding(b))   # [B, T_b, E_dim]

        a_ = lstm_compare(a_emb)          # [B, T_a, H*2]
        b_ = lstm_compare(b_emb)          # [B, T_b, H*2]

        # mask a_ and b_, since the <pad> position is no more zero
        a_ = keras.layers.Multiply()([a_, self._expand_dim(a_mask, axis=2)])
        b_ = keras.layers.Multiply()([b_, self._expand_dim(b_mask, axis=2)])

        # local inference
        e = keras.layers.Dot(axes=-1)([a_, b_])   # [B, T_a, T_b]
        _ab_mask = keras.layers.Multiply()(       # _ab_mask: [B, T_a, T_b]
            [self._expand_dim(a_mask, axis=2),    # [B, T_a, 1]
             self._expand_dim(b_mask, axis=1)])   # [B, 1, T_b]

        pm = keras.layers.Permute((2, 1))
        mask_layer = self._make_atten_mask_layer()
        softmax_layer = keras.layers.Softmax(axis=-1)

        e_a = softmax_layer(mask_layer([e, _ab_mask]))          # [B, T_a, T_b]
        e_b = softmax_layer(mask_layer([pm(e), pm(_ab_mask)]))  # [B, T_b, T_a]

        # alignment (a_t = a~, b_t = b~ )
        a_t = keras.layers.Dot(axes=(2, 1))([e_a, b_])   # [B, T_a, H*2]
        b_t = keras.layers.Dot(axes=(2, 1))([e_b, a_])   # [B, T_b, H*2]

        # local inference info enhancement
        m_a = keras.layers.Concatenate(axis=-1)([
            a_,
            a_t,
            keras.layers.Subtract()([a_, a_t]),
            keras.layers.Multiply()([a_, a_t])])    # [B, T_a, H*2*4]
        m_b = keras.layers.Concatenate(axis=-1)([
            b_,
            b_t,
            keras.layers.Subtract()([b_, b_t]),
            keras.layers.Multiply()([b_, b_t])])    # [B, T_b, H*2*4]

        # project m_a and m_b from 4*H*2 dim to H dim
        m_a = dropout(dense_compare(m_a))   # [B, T_a, H]
        m_b = dropout(dense_compare(m_b))   # [B, T_a, H]

        # inference composition
        v_a = lstm_compose(m_a)          # [B, T_a, H*2]
        v_b = lstm_compose(m_b)          # [B, T_b, H*2]

        # pooling
        v_a = keras.layers.Concatenate(axis=-1)(
            [self._avg(v_a, a_mask), self._max(v_a, a_mask)])   # [B, H*4]
        v_b = keras.layers.Concatenate(axis=-1)(
            [self._avg(v_b, b_mask), self._max(v_b, b_mask)])   # [B, H*4]
        v = keras.layers.Concatenate(axis=-1)([v_a, v_b])       # [B, H*8]

        # mlp (multilayer perceptron) classifier
        output = self._make_multi_layer_perceptron_layer()(v)  # [B, H]
        output = dropout(output)
        output = self._make_output_layer()(output)             # [B, #classes]

        self._backend = keras.Model(inputs=[a, b], outputs=output)
