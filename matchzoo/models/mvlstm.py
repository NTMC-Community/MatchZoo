"""An implementation of MVLSTM Model."""
import typing

import keras
import keras.backend as K

import matchzoo
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class MVLSTM(BaseModel):
    """
    MVLSTM Model.

    Examples:
        >>> model = MVLSTM()
        >>> model.params['lstm_units'] = 32
        >>> model.params['top_k'] = 50
        >>> model.params['mlp_num_layers'] = 2
        >>> model.params['mlp_num_units'] = 20
        >>> model.params['mlp_num_fan_out'] = 10
        >>> model.params['mlp_activation_func'] = 'relu'
        >>> model.params['dropout_rate'] = 0.5
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True, with_multi_layer_perceptron=True)
        params.add(Param(name='lstm_units', value=32,
                         desc="Integer, the hidden size in the "
                              "bi-directional LSTM layer."))
        params.add(Param(name='dropout_rate', value=0.0,
                         desc="Float, the dropout rate."))
        params.add(Param(
            'top_k', value=10,
            hyper_space=hyper_spaces.quniform(low=2, high=100),
            desc="Integer, the size of top-k pooling layer."
        ))
        params['optimizer'] = 'adam'
        return params

    def build(self):
        """Build model structure."""
        query, doc = self._make_inputs()

        # Embedding layer
        embedding = self._make_embedding_layer()
        embed_query = embedding(query)
        embed_doc = embedding(doc)

        # Bi-directional LSTM layer
        rep_query = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))(embed_query)
        rep_doc = keras.layers.Bidirectional(keras.layers.LSTM(
            self._params['lstm_units'],
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))(embed_doc)

        # Top-k matching layer
        matching_matrix = keras.layers.Dot(
            axes=[2, 2], normalize=False)([rep_query, rep_doc])
        matching_signals = keras.layers.Reshape((-1,))(matching_matrix)
        matching_topk = keras.layers.Lambda(
            lambda x: K.tf.nn.top_k(x, k=self._params['top_k'], sorted=True)[0]
        )(matching_signals)

        # Multilayer perceptron layer.
        mlp = self._make_multi_layer_perceptron_layer()(matching_topk)
        mlp = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp)

        x_out = self._make_output_layer()(mlp)
        self._backend = keras.Model(inputs=[query, doc], outputs=x_out)
