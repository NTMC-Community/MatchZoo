"""HBMP model."""
import keras
import typing

from matchzoo.engine import hyper_spaces
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel


class HBMP(BaseModel):
    """
    HBMP model.

    Examples:
        >>> model = HBMP()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.params['embedding_input_dim'] = 200
        >>> model.params['embedding_output_dim'] = 100
        >>> model.params['embedding_trainable'] = True
        >>> model.params['alpha'] = 0.1
        >>> model.params['mlp_num_layers'] = 3
        >>> model.params['mlp_num_units'] = [10, 10]
        >>> model.params['lstm_num_units'] = 5
        >>> model.params['dropout_rate'] = 0.1
        >>> model.build()
    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'
        params.add(Param(name='alpha', value=0.1,
                         desc="Negative slope coefficient of LeakyReLU "
                              "function."))
        params.add(Param(name='mlp_num_layers', value=3,
                         desc="The number of layers of mlp."))
        params.add(Param(name='mlp_num_units', value=[10, 10],
                         desc="The hidden size of the FC layers, but not "
                              "include the final layer."))
        params.add(Param(name='lstm_num_units', value=5,
                         desc="The hidden size of the LSTM layer."))
        params.add(Param(
            name='dropout_rate', value=0.1,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        """Build model structure."""
        input_left, input_right = self._make_inputs()

        embedding = self._make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # Get sentence embedding
        embed_sen_left = self._sentence_encoder(
            embed_left,
            lstm_num_units=self._params['lstm_num_units'],
            drop_rate=self._params['dropout_rate'])
        embed_sen_right = self._sentence_encoder(
            embed_right,
            lstm_num_units=self._params['lstm_num_units'],
            drop_rate=self._params['dropout_rate'])

        # Concatenate two sentence embedding: [embed_sen_left, embed_sen_right,
        # |embed_sen_left-embed_sen_right|, embed_sen_left*embed_sen_right]
        embed_minus = keras.layers.Subtract()(
            [embed_sen_left, embed_sen_right])
        embed_minus_abs = keras.layers.Lambda(lambda x: abs(x))(embed_minus)
        embed_multiply = keras.layers.Multiply()(
            [embed_sen_left, embed_sen_right])
        concat = keras.layers.Concatenate(axis=1)(
            [embed_sen_left, embed_sen_right, embed_minus_abs, embed_multiply])

        # Multiply perception layers to classify
        mlp_out = self._classifier(
            concat,
            mlp_num_layers=self._params['mlp_num_layers'],
            mlp_num_units=self._params['mlp_num_units'],
            drop_rate=self._params['dropout_rate'],
            leaky_relu_alpah=self._params['alpha'])
        out = self._make_output_layer()(mlp_out)

        self._backend = keras.Model(
            inputs=[input_left, input_right], outputs=out)

    def _classifier(
        self,
        input_: typing.Any,
        mlp_num_layers: int,
        mlp_num_units: list,
        drop_rate: float,
        leaky_relu_alpah: float
    ) -> typing.Any:
        for i in range(mlp_num_layers - 1):
            input_ = keras.layers.Dropout(rate=drop_rate)(input_)
            input_ = keras.layers.Dense(mlp_num_units[i])(input_)
            input_ = keras.layers.LeakyReLU(alpha=leaky_relu_alpah)(input_)

        return input_

    def _sentence_encoder(
        self,
        input_: typing.Any,
        lstm_num_units: int,
        drop_rate: float
    ) -> typing.Any:
        """
        Stack three BiLSTM MaxPooling blocks as a hierarchical structure.
        Concatenate the output of three blocs as the input sentence embedding.
        Each BiLSTM layer reads the input sentence as the input.
        Each BiLSTM layer except the first one is initialized(the initial
        hidden state and the cell state) with the final state of the previous
        layer.
        """
        emb1 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=lstm_num_units,
                return_sequences=True,
                return_state=True,
                dropout=drop_rate,
                recurrent_dropout=drop_rate),
            merge_mode='concat')(input_)
        emb1_maxpooling = keras.layers.GlobalMaxPooling1D()(emb1[0])

        emb2 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=lstm_num_units,
                return_sequences=True,
                return_state=True,
                dropout=drop_rate,
                recurrent_dropout=drop_rate),
            merge_mode='concat')(input_, initial_state=emb1[1:5])
        emb2_maxpooling = keras.layers.GlobalMaxPooling1D()(emb2[0])

        emb3 = keras.layers.Bidirectional(
            keras.layers.LSTM(
                units=lstm_num_units,
                return_sequences=True,
                return_state=True,
                dropout=drop_rate,
                recurrent_dropout=drop_rate),
            merge_mode='concat')(input_, initial_state=emb2[1:5])
        emb3_maxpooling = keras.layers.GlobalMaxPooling1D()(emb3[0])

        emb = keras.layers.Concatenate(axis=1)(
            [emb1_maxpooling, emb2_maxpooling, emb3_maxpooling])

        return emb
