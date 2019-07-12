"""BiMPM."""

from keras.models import Model
from keras.layers import Dense, Concatenate, Dropout
from keras.layers import Bidirectional, LSTM

from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.base_model import BaseModel
from matchzoo.contrib.layers import MultiPerspectiveLayer


class BiMPM(BaseModel):
    """
    BiMPM.

    Reference:
    https://github.com/zhiguowang/BiMPM/blob/master/src/SentenceMatchModelGraph.py#L43-L186
    Examples:
        >>> import matchzoo as mz
        >>> model = mz.contrib.models.BiMPM()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'

        # params.add(Param('dim_word_embedding', 50))
        # TODO(tjf): remove unused params in the final version
        # params.add(Param('dim_char_embedding', 50))
        # params.add(Param('word_embedding_mat'))
        # params.add(Param('char_embedding_mat'))
        # params.add(Param('embedding_random_scale', 0.2))
        # params.add(Param('activation_embedding', 'softmax'))

        # BiMPM Setting
        params.add(Param('perspective', {'full': True,
                                         'max-pooling': True,
                                         'attentive': True,
                                         'max-attentive': True}))
        params.add(Param('mp_dim', 3))
        params.add(Param('att_dim', 3))
        params.add(Param('hidden_size', 4))
        params.add(Param('dropout_rate', 0.0))
        params.add(Param('w_initializer', 'glorot_uniform'))
        params.add(Param('b_initializer', 'zeros'))
        params.add(Param('activation_hidden', 'linear'))

        params.add(Param('with_match_highway', False))
        params.add(Param('with_aggregation_highway', False))

        return params

    def build(self):
        """Build model structure."""
        # ~ Input Layer
        input_left, input_right = self._make_inputs()

        # Word Representation Layer
        # TODO: concatenate word level embedding and character level embedding.
        embedding = self._make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # L119-L121
        # https://github.com/zhiguowang/BiMPM/blob/master/src/SentenceMatchModelGraph.py#L119-L121
        embed_left = Dropout(self._params['dropout_rate'])(embed_left)
        embed_right = Dropout(self._params['dropout_rate'])(embed_right)

        # ~ Word Level Matching Layer
        # Reference:
        # https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py#L207-L223
        # TODO
        pass

        # ~ Encoding Layer
        # Note: When merge_mode = None, output will be [forward, backward],
        # The default merge_mode is concat, and the output will be [lstm].
        # If with return_state, then the output would append [h,c,h,c].
        bi_lstm = Bidirectional(
            LSTM(self._params['hidden_size'],
                 return_sequences=True,
                 return_state=True,
                 dropout=self._params['dropout_rate'],
                 kernel_initializer=self._params['w_initializer'],
                 bias_initializer=self._params['b_initializer']),
            merge_mode='concat')
        # x_left = [lstm_lt, forward_h_lt, _, backward_h_lt, _ ]
        x_left = bi_lstm(embed_left)
        x_right = bi_lstm(embed_right)

        # ~ Multi-Perspective Matching layer.
        # Output is two sequence of vectors.
        # Cons: Haven't support multiple context layer
        multi_perspective = MultiPerspectiveLayer(self._params['att_dim'],
                                                  self._params['mp_dim'],
                                                  self._params['perspective'])
        # Note: input to `keras layer` must be list of tensors.
        mp_left = multi_perspective(x_left + x_right)
        mp_right = multi_perspective(x_right + x_left)

        # ~ Dropout Layer
        mp_left = Dropout(self._params['dropout_rate'])(mp_left)
        mp_right = Dropout(self._params['dropout_rate'])(mp_right)

        # ~ Highway Layer
        # reference:
        # https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py#L289-L295
        if self._params['with_match_highway']:
            # the input is left matching representations (question / passage)
            pass

        # ~ Aggregation layer
        # TODO: mask the above layer
        aggregation = Bidirectional(
            LSTM(self._params['hidden_size'],
                 return_sequences=False,
                 return_state=False,
                 dropout=self._params['dropout_rate'],
                 kernel_initializer=self._params['w_initializer'],
                 bias_initializer=self._params['b_initializer']),
            merge_mode='concat')
        rep_left = aggregation(mp_left)
        rep_right = aggregation(mp_right)

        # Concatenate the concatenated vector of left and right.
        x = Concatenate()([rep_left, rep_right])

        # ~ Highway Network
        # reference:
        # https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py#L289-L295
        if self._params['with_aggregation_highway']:
            pass

        # ~ Prediction layer.
        # reference:
        # https://github.com/zhiguowang/BiMPM/blob/master/src/SentenceMatchModelGraph.py#L140-L153
        x = Dense(self._params['hidden_size'],
                  activation=self._params['activation_hidden'])(x)
        x = Dense(self._params['hidden_size'],
                  activation=self._params['activation_hidden'])(x)
        x_out = self._make_output_layer()(x)
        self._backend = Model(inputs=[input_left, input_right],
                              outputs=x_out)
