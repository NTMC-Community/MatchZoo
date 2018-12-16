"""An implementation of aNMM Model."""


import logging

import keras
from keras.activations import softmax
from keras.initializers import RandomUniform

from matchzoo import engine

logger = logging.getLogger(__name__)


class ANMMModel(engine.BaseModel):
    """
    ANMM Model.

    Examples:
        >>> model = ANMMModel()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(engine.Param('bin_num', 60))
        params.add(engine.Param('dropout_rate', 0.1))
        params.add(engine.Param('left_text_len', 10))
        params.add(engine.Param('num_layers', 2))
        params.add(engine.Param('hidden_sizes', [5, 10]))
        return params

    def build(self):
        """
        Build model structure.

        aNMM model based on bin weighting and query term attentions
        """
        # query is [batch_size, left_text_len]
        # doc is [batch_size, right_text_len, bin_num]
        query, doc = self._make_inputs()
        embedding = self._make_embedding_layer()

        q_embed = embedding(query)
        q_attention = keras.layers.Dense(
            1, kernel_initializer=RandomUniform(), use_bias=False)(q_embed)
        q_attention = keras.layers.Lambda(
            lambda x: softmax(x, axis=1),
            output_shape=(self._params['left_text_len'],)
        )(q_attention)
        d_bin = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(doc)
        for layer_id in range(self._params['num_layers'] - 1):
            d_bin = keras.layers.Dense(
                self._params['hidden_sizes'][layer_id],
                kernel_initializer=RandomUniform())(d_bin)
            d_bin = keras.layers.Activation('tanh')(d_bin)
            d_bin = keras.layers.Dense(
                self._params['hidden_sizes'][self._params['num_layers'] - 1])(
                d_bin)
            d_bin = keras.layers.Reshape((self._params['left_text_len'],))(
                d_bin)
        q_attention = keras.layers.Reshape((self._params['left_text_len'],))(
            q_attention)
        out = keras.layers.Dot(axes=[1, 1])([d_bin, q_attention])
        self._backend = keras.Model(inputs=[query, doc], outputs=out)
