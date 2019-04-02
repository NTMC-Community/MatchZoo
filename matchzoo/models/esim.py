"""KNRM model."""
import keras
import tensorflow as tf
from keras.layers import * # Dense, Dot, Softmax, Concatenate, Embedding, LSTM, Bidirectional
import keras.backend as K

import matchzoo as mz
from matchzoo import engine, preprocessors


class ESIM(engine.BaseModel):
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

        >>> model.params['mlp_num_layers'] = 0      # zero hidden layer (one output layer)
        >>> model.params['mlp_num_fan_out'] = 300
        >>> model.params['mlp_activation_func'] = 'tanh'

        >>> model.params['mask_value'] = 0        
        >>> model.params['dropout_rate'] = 0.5

        >>> model.guess_and_fill_missing_params()
        >>> model.build()
    """

    @classmethod
    def get_default_params(cls)  -> engine.ParamTable:
        """Get default parameters."""
        params = super().get_default_params(with_embedding=True,
                                            with_multi_layer_perceptron=True)

        params.add(engine.Param(
            name='dropout_rate',
            value=0.5,
            desc="The dropout rate for all fully-connected layer"
        ))

        params.add(engine.Param(
            name='lstm_dim',
            value=300,
            desc="The dimension of LSTM layer."
        ))

        params.add(engine.Param(
            name='mask_value',
            value=0,
            desc="The value would be regarded as pad"
        ))

        return params

    def compile(self, optimizer=None):
        """
        Overwrite the engine.BaseModel.compile() to allow self-defined optimizer and learning rate

        Examples:
            >>> from matchzoo import models
            >>> model = models.Naive()
            >>> model.guess_and_fill_missing_params(verbose=0)
            >>> model.params['task'].metrics = ['mse', 'map']
            >>> model.params['task'].metrics
            ['mse', mean_average_precision(0.0)]
            >>> model.build()
            >>> model.compile()
        """
        if optimizer == None:
            optimizer = self._params['optimizer']

        self._backend.compile(optimizer=optimizer,
                              loss=self._params['task'].loss)


    def _make_inputs(self) -> list:
        """
        prepare input from input dict
        """        
        input_left = keras.layers.Input(
            name='text_left',
            shape=self._params['input_shapes'][0]
        )
        input_right = keras.layers.Input(
            name='text_right',
            shape=self._params['input_shapes'][1]
        )
        return [input_left, input_right]

    
    def _make_embedding_layer(self, name: str = 'embedding') -> keras.layers.Layer:
        """
        Overwrite the engine.BaseModel._make_embedding_layer to allow specifiying mask_zero
        """
        return keras.layers.Embedding(
            self._params['embedding_input_dim'],
            self._params['embedding_output_dim'],
            trainable=self._params['embedding_trainable'],
            mask_zero = True,
            name=name
        )


    def _expand_dim(self, inp, axis):
        """
        Wrap keras.backend.expand_dims into a Lambda layer
        """
        return keras.layers.Lambda(lambda x: K.expand_dims(x, axis=axis))(inp)


    def _make_atten_mask_layer(self) -> keras.layers.Layer:
        """
        make mask layer for attention weight matrix so that each word won't pay attention to <PAD> timestep
        """
        return keras.layers.Lambda(
            lambda weight_mask: weight_mask[0] + (1.0 - weight_mask[1]) * -1e7,
            name="atten_mask"
        )


    def _make_BiLSTM_layer(self, lstm_dim) -> keras.layers.Layer:
        """
        Bidirectional LSTM layer in ESIM.

        :param lstm_dim: int, dimension of LSTM layer
        :return: `keras.layers.Layer`.
        """
        return Bidirectional(layer=LSTM(lstm_dim, return_sequences=True),
                             merge_mode='concat')


    def _max(self, texts, mask):
        """
        Compute the max of each text according to their real length

        :param texts: np.array with shape [B, T, H]
        :param lengths: np.array with shape [B, T, ], where 1 means valid, 0 means pad
        """
        mask = self._expand_dim(mask, axis=2)
        new_texts = Multiply()([texts, mask])

        text_max = Lambda(
            lambda x: K.max(x, axis=1),
        )(new_texts)

        return text_max


    def _avg(self, texts, mask):
        """
        Compute the mean of each text according to their real length

        :param texts: np.array with shape [B, T, H]
        :param lengths: np.array with shape [B, T, ], where 1 means valid, 0 means pad
        """
        mask = self._expand_dim(mask, axis=2)
        new_texts = Multiply()([texts, mask])

        # timestep-wise division, exclude the PAD number when calc avg
        text_avg = keras.layers.Lambda(
            lambda text_mask: K.sum(text_mask[0], axis=1) / K.sum(text_mask[1], axis=1),
        )([new_texts, mask])

        return text_avg

    def build(self):
        """Build model."""

        # parameters
        lstm_dim = self._params['lstm_dim']
        dropout_rate = self._params['dropout_rate']

        # layers
        create_mask = Lambda(lambda x: K.cast(K.not_equal(x, self._params['mask_value']), K.floatx()))
        embedding = self._make_embedding_layer()
        lstm_compare = self._make_BiLSTM_layer(lstm_dim)
        lstm_compose = self._make_BiLSTM_layer(lstm_dim)
        dense_compare = Dense(units=lstm_dim, activation='relu', use_bias=True)
        dropout = Dropout(dropout_rate)


        # model
        a, b = self._make_inputs()      # [B, T_a], [B, T_b]
        a_mask = create_mask(a)         # [B, T_a]
        b_mask = create_mask(b)         # [B, T_b]

        ########################
        # encoding
        ########################
        a_emb = dropout(embedding(a))   # [B, T_a, E_dim]
        b_emb = dropout(embedding(b))   # [B, T_b, E_dim]

        a_ = lstm_compare(a_emb)          # [B, T_a, H*2]
        b_ = lstm_compare(b_emb)          # [B, T_b, H*2]


        ########################
        # local inference
        ########################
        # similarity matrix
        e = Dot(axes=-1)([a_, b_])  # [B, T_a, T_b]
        _ab_mask = Multiply()([self._expand_dim(a_mask, axis=2),    # [B, T_a, 1]
                               self._expand_dim(b_mask, axis=1)])   # [B, 1, T_b]
        # _ab_mask: [B, T_a, T_b]

        pm = Permute((2, 1))
        mask_layer = self._make_atten_mask_layer()
        softmax_layer = Softmax(axis=-1)

        e_a = softmax_layer(mask_layer([e, _ab_mask]))          # [B, T_a, T_b]
        e_b = softmax_layer(mask_layer([pm(e), pm(_ab_mask)]))  # [B, T_b, T_a]

        # alignment (a_t = a~, b_t = b~ )
        a_t = Dot(axes=(2, 1))([e_a, b_])   # [B, T_a, H*2]
        b_t = Dot(axes=(2, 1))([e_b, a_])   # [B, T_b, H*2]

        # local inference info enhancement
        m_a = Concatenate(axis=-1)([a_, a_t, Subtract()([a_, a_t]), Multiply()([a_, a_t])])    # [B, T_a, H*2*4]
        m_b = Concatenate(axis=-1)([b_, b_t, Subtract()([b_, b_t]), Multiply()([b_, b_t])])    # [B, T_b, H*2*4]

        # project m_a and m_b from 4*H*2 dim to H dim
        m_a = dropout(dense_compare(m_a))   # [B, T_a, H]
        m_b = dropout(dense_compare(m_b))   # [B, T_a, H]


        ########################
        # inference composition
        ########################
        v_a = lstm_compose(m_a)          # [B, T_a, H*2]
        v_b = lstm_compose(m_b)          # [B, T_b, H*2]

        # pooling
        v_a = Concatenate(axis=-1)([self._avg(v_a, a_mask), self._max(v_a, a_mask)]) # [B, H*4]
        v_b = Concatenate(axis=-1)([self._avg(v_b, b_mask), self._max(v_b, b_mask)]) # [B, H*4]
        v = Concatenate(axis=-1)([v_a, v_b])                                         # [B, H*8]

        # mlp (multilayer perceptron) classifier
        output = dropout(self._make_multi_layer_perceptron_layer()(v))               # [B, H]
        output = self._make_output_layer()(output)                                   # [B, #classes]

        self._backend = keras.Model(inputs=[a, b], outputs=output)


