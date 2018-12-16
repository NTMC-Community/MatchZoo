"""An implementation of MVLSTM Model."""
import typing

import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, Embedding, Dropout, LSTM, Dot, \
    Reshape, Lambda

from matchzoo import engine


class MVLSTMModel(engine.BaseModel):
    """
    MVLSTM Model.
    Examples:
        >>> model = MVLSTMModel()
        >>> model.guess_and_fill_missing_params()
        >>> model.build()
    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'
        params['input_shapes'] = [(32,), (32,)]
        params.add(engine.Param('trainable_embedding', False))
        params.add(engine.Param('vocab_size', 100))
        params.add(engine.Param('hidden_size', 32))
        params.add(engine.Param('padding', 'same'))
        params.add(engine.Param('dropout_rate', 0.0))
        params.add(engine.Param(
            'top_k', value=10,
            hyper_space=engine.hyper_spaces.quniform(low=2, high=100)
        ))
        return params


    def build(self):
        """Build model structure."""

        query, doc = self._make_inputs()

        embedding = self._make_embedding_layer()
        embed_left = embedding(query)
        embed_right = embedding(doc)
        # Process query & document input.
        bi_lstm = Bidirectional(LSTM(self._params['hidden_size'],
                              return_sequences=True,
                              dropout=self._params['dropout_rate']))

        rep_left = bi_lstm(embed_left)
        rep_right = bi_lstm(embed_right)
        cross = Dot(axes=[1, 1], normalize=False)([rep_left, rep_right])
        cross = Reshape((-1, ))(cross)

        mm_k = Lambda(lambda x: K.tf.nn.top_k(x, 
                                              k=self._params['top_k'],
                                              sorted=True)[0]
                                              )(cross)
        mm_k = Dropout(rate=self._params['dropout_rate'])(mm_k)

        x_out = self._make_output_layer()(mm_k)
        self._backend = Model(
            inputs=[query, doc],
            outputs=x_out)
