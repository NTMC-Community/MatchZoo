"""Naive model with a simplest structure for testing purposes."""

import numpy as np
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, Dense, Concatenate, Dropout


from matchzoo import engine
from matchzoo import preprocessors
from matchzoo.layers import MultiPerspectiveLayer


class BimpmModel(engine.BaseModel):
    """Naive model with a simplest structure for testing purposes."""

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'

        params.add(engine.Param('dim_word_embedding', 50))
        params.add(engine.Param('dim_char_embedding', 50))
        params.add(engine.Param('word_embedding_mat', None))
        params.add(engine.Param('char_embedding_mat', None))
        params.add(engine.Param('embedding_random_scale', 0.2))
        params.add(engine.Param('activation_embedding', 'softmax'))
        # Bimpm Setting
        params.add(engine.Param('perspective', {'full': True,
                                                'max-pooling': True,
                                                'attentive': True,
                                                'max-attentive': True}))
        params.add(engine.Param('dim_perspective', 20))
        params.add(engine.Param('hidden_size', 128))
        params.add(engine.Param('dropout_rate', 0.0))
        params.add(engine.Param('w_initializer', 'glorot_uniform'))
        params.add(engine.Param('b_initializer', 'zeros'))
        params.add(engine.Param('activation_hidden', 'linear'))

        return params

    @classmethod
    def get_default_preprocessor(cls):
        """:return: Instance of :class:`NaivePreprocessor`."""
        return preprocessors.NaivePreprocessor()

    @property
    def word_embedding_mat(self) -> np.ndarray:
        """Get pretrained embedding for ArcI model."""
        # Check if provided embedding matrix
        if self._params['word_embedding_mat'] is None:
            raise TypeError('No pre-trained word embeddings provided.')

    @word_embedding_mat.setter
    def word_embedding_mat(self, embedding_mat: np.ndarray):
        """
        Set pretrained embedding for ArcI model.

        :param embedding_mat: pretrained embedding in numpy format.
        """
        self._params['word_embedding_mat'] = embedding_mat
        self._params['vocab_size'], self._params['dim_word_embedding'] = \
            embedding_mat.shape

    @property
    def char_embedding_mat(self) -> np.ndarray:
        """Initialize character level embedding."""
        s = self._params['embedding_random_scale']
        self._params['char_embedding_mat'] = \
            np.random.uniform(-s, s, (self._params['vocab_size'],
                                      self._params['dim_char_embedding']))
        return self._params['char_embedding_mat']

    def char_embedding(self, char_input_shape, char_vocab_size):
        """Create a character level embedding model."""
        input_char = Input(shape=char_input_shape)
        embed_char = LSTM(self._params['dim_char_embedding'],
                          kernel_initializer=self._params['w_initializer'],
                          bias_initializer=self._params['b_initializer'])(
                              input_char)
        embed_char = Dense(char_vocab_size,
                           activation=self._params['activation_embedding'])(
                               embed_char)
        return Model(input_char, embed_char)

    def build(self):
        """
        Build model structure
        """
        # ~ Input Layer
        input_left, input_right = self._make_inputs()

        # Word Representation Layer
        # TODO: concatenate word level embedding and character level embedding.
        embedding = self._make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # ~ Encoding Layer
        # Note: When merge_mode = None, output will be [lstm_forward, lstm_backward],
        #       the default setting of merge_mode is concat, and the output will be
        #       [lstm]. If with return_state, then the output would append [h,c,h,c]
        bi_lstm = Bidirectional(LSTM(self._params['hidden_size'],
                                     return_sequences=True,
                                     return_state=True,
                                     dropout=self._params['dropout_rate'],
                                     kernel_initializer=self._params['w_initializer'],
                                     bias_initializer=self._params['b_initializer']),
                                merge_mode='concat')
        # x_left = [lstm_lt, forward_h_lt, _, backward_h_lt, _ ]
        x_left = bi_lstm(embed_left)
        x_right = bi_lstm(embed_right)

        # ~ Word Level Matching Layer
        # Reference: https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py#L207-L223
        # TODO pass

        # ~ Multi-Perspective Matching layer.
        # Output is two sequence of vectors.
        # Cons: Haven't support multiple context layer
        multi_perspective = MultiPerspectiveLayer(dim_input=self._params['hidden_size']*2,
                                                  dim_perspective=self._params['dim_perspective'],
                                                  perspective=self._params['perspective'])
        # Note: input to `keras layer` must be list of tensors.
        mp_left = multi_perspective(x_left + x_right)
        mp_right = multi_perspective(x_right + x_left)

        # ~ Dropout Layer
        mp_left = Dropout(self._params['dropout_rate'])(mp_left)
        mp_right = Dropout(self._params['dropout_rate'])(mp_right)

        # ~ Highway Layer
        # reference: https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py#L289-L295
        if self.with_match_highway:
            # the input is left matching representations (question / passage)
            pass

        # ~ Aggregation layer
        # TODO: before input into the Bidirectional layer, the above layer has been masked
        aggregation = Bidirectional(LSTM(self._params['hidden_size'],
                                         return_sequences=False,
                                         return_state=False,
                                         dropout=self._params['dropout_rate'],
                                         kernel_initializer=self._params['w_initializer'],
                                         bias_initializer=self._params['b_initializer']),
                                    merge_mode='concat')
        rep_left = aggregation(mp_left)
        rep_right = aggregation(mp_right)

        # ~ Highway Network
        # reference: https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py#L289-L295
        if self.with_aggregation_highway:
            pass

        # Concatenate the concatenated vector of left and right.
        # TODO(tjf) add more functions here
        # ref: https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py#L206-L339
        x = Concatenate()([rep_left, rep_right])

        # Prediction layer.
        x = Dense(self._params['hidden_size'],
                  activation=self._params['activation_hidden'])(x)
        x = Dense(self._params['hidden_size'],
                  activation=self._params['activation_hidden'])(x)
        x_out = self._make_output_layer()(x)
        self._backend = Model(inputs=[input_left, input_right],
                              outputs=x_out)
