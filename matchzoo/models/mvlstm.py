"""An implementation of MVLSTM Model."""
import typing
import logging

import keras
import keras.backend as K

from matchzoo import engine

logger = logging.getLogger(__name__)


class MVLSTM(engine.BaseModel):
    """
    MVLSTM Model.

    Examples:
        >>> model = MVLSTM()
        >>> model.params['mlp_num_layers'] = 1
        >>> model.params['top_k'] = 10
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True, with_multi_layer_perceptron=True)
        params['optimizer'] = 'adam'
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
        embed_query = embedding(query)
        embed_doc = embedding(doc)
        # Process query & document input.
        bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['hidden_size'],
            return_sequences=True, dropout=self._params['dropout_rate']))

        rep_query = bi_lstm(embed_query)
        rep_doc = bi_lstm(embed_doc)

        matching_matrix = keras.layers.Dot(
            axes=[1, 1], normalize=False)([rep_query, rep_doc])
        matching_matrix = keras.layers.Reshape((-1, ))(matching_matrix)

        matching_topk = keras.layers.Lambda(
            lambda x: K.tf.nn.top_k(x, k=self._params['top_k'], sorted=True)[0]
        )(matching_matrix)
        mlp = self._make_multi_layer_perceptron_layer()(matching_topk)
        mlp = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp)

        x_out = self._make_output_layer()(mlp)
        self._backend = keras.Model(inputs=[query, doc], outputs=x_out)
