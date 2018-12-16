"""An implementation of MVLSTM Model."""
import typing

import keras
import keras.backend as K

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
        params['input_shapes'] = [(5,), (300,)]
        params['embedding_output_dim'] =  50
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
        embed_query = embedding(query)
        embed_doc = embedding(doc)
        # Process query & document input.
        bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['hidden_size'],
            return_sequences=True,dropout=self._params['dropout_rate']))

        rep_query = bi_lstm(embed_query)
        rep_doc = bi_lstm(embed_doc)

        matching_matrix = keras.layers.Dot(
            axes=[1, 1], normalize=False)([rep_query, rep_doc])
        matching_matrix = keras.layers.Reshape((-1, ))(matching_matrix)

        matching_topk = keras.layers.Lambda(
            lambda x: K.tf.nn.top_k(x, k=self._params['top_k'], sorted=True)[0]
        )(matching_matrix)
        matching_topk = keras.layers.Dropout(
                    rate=self._params['dropout_rate'])(matching_topk)

        x_out = self._make_output_layer()(matching_topk)
        self._backend = keras.Model(inputs=[query, doc], outputs=x_out)
