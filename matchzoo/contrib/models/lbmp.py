"""LBMP model."""
import typing
import keras

from matchzoo.engine import hyper_spaces
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel


class LBMP(BaseModel):
    """
    LBMP model.

    Examples:
        >>> model = LBMP()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.params['embedding_input_dim'] = 10000
        >>> model.params['embedding_output_dim'] = 100
        >>> model.params['embedding_trainable'] = True
        >>> model.params['leaky_relu_alpha'] = 0.1
        >>> model.params['mlp_num_layers'] = 3
        >>> model.params['mlp_num_units'] = [600, 600]
        >>> model.params['lstm_num_units'] = 600
        >>> model.params['dropout_rate'] = 0.1
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'
        params.add(Param(name='leaky_relu_alpha', value=0.1,
                         desc="Negative slope coefficient of LeakyReLU function."))
        params.add(Param(name='mlp_num_layers', value=3,
                         desc="The number of layers of mlp."))
        params.add(Param(name='mlp_num_units', value=[600, 600],
                         desc="The hidden size of the FC layers, but not include the final layer."))
        params.add(Param(name='lstm_num_units', value=600,
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
      
        # get sentence embedding
        embed_sen_left = self._sentence_encoder(
            embed_left, lstm_num_units=self._params['lstm_num_units'], drop_rate=self._params['dropout_rate'])
        embed_sen_right = self._sentence_encoder(
            embed_right, lstm_num_units=self._params['lstm_num_units'], drop_rate=self._params['dropout_rate'])

        # concatenate two sentence embedding: [embed_sen_left, embed_sen_right, |embed_sen_left-embed_sen_right|, embed_sen_left*embed_sen_right]
        concat1 = keras.layers.Subtract()([embed_sen_left, embed_sen_right])
        concat1_abs = keras.layers.Lambda(lambda x: abs(x))(concat1)
        concat2 = keras.layers.Multiply()([embed_sen_left, embed_sen_right])
        concat = keras.layers.Concatenate(axis=1)(
            [embed_sen_left, embed_sen_right, concat1_abs, concat2])

        # multiply perception layers to classify
        out = self._classifier(concat, mlp_num_layers=self._params['mlp_num_layers'], mlp_num_units=self._params[
                               'mlp_num_units'], drop_rate=self._params['dropout_rate'], leaky_relu_alpah=self._params['leaky_relu_alpha'])

        self._backend = keras.Model(
            inputs=[input_left, input_right], outputs=out)

    def _classifier(
        self,
        input_: typing.Any,
        mlp_num_layers: int,
        mlp_num_units: list,
        drop_rate: float,
        leaky_relu_alpah: float
    )-> typing.Any:
        for i in range(mlp_num_layers - 1):
            input_ = keras.layers.Dropout(rate=drop_rate)(input_)  
            input_ = keras.layers.Dense(mlp_num_units[i])(input_)
            input_ = keras.layers.LeakyReLU(alpha=leaky_relu_alpah)(input_)

        return self._make_output_layer()(input_)


    def _sentence_encoder(
        self,
        input_: typing.Any,
        lstm_num_units: int,
        drop_rate: float
    ) -> typing.Any:
        """
        stack three BiLSTM MaxPooling blocks as a hierarchical structure.
        concatenate the output of three blocs as the input sentence embedding.
        each BiLSTM layer reads the input sentence as the input.
        each BiLSTM layer except the first one is initialized(the initial hidden state and the cell state) with the final state of the previous layer.   
        """
        layer1_fw, h1_fw, c1_fw = keras.layers.LSTM(
            units=lstm_num_units, return_sequences=True, return_state=True, dropout=drop_rate, recurrent_dropout=drop_rate)(input_)
        layer1_bw, h1_bw, c1_bw = keras.layers.LSTM(
            units=lstm_num_units, return_sequences=True, return_state=True, dropout=drop_rate, recurrent_dropout=drop_rate, go_backwards=True)(input_)
        layer1 = keras.layers.Concatenate(axis=2)([layer1_fw, layer1_bw])
        layer1_maxpooling = keras.layers.GlobalMaxPooling1D()(layer1)

        layer2_fw, h2_fw, c2_fw = keras.layers.LSTM(units=lstm_num_units, return_sequences=True, return_state=True,
                                                    dropout=drop_rate, recurrent_dropout=drop_rate)(input_, initial_state=[c1_fw, h1_fw])
        layer2_bw, h2_bw, c2_bw = keras.layers.LSTM(units=lstm_num_units, return_sequences=True, return_state=True,
                                                    dropout=drop_rate, recurrent_dropout=drop_rate, go_backwards=True)(input_, initial_state=[c1_bw, h1_bw])
        layer2 = keras.layers.Concatenate(axis=2)([layer2_fw, layer2_bw])
        layer2_maxpooling = keras.layers.GlobalMaxPooling1D()(layer2)

        layer3_fw = keras.layers.LSTM(units=lstm_num_units, return_sequences=True, dropout=drop_rate,
                                      recurrent_dropout=drop_rate)(input_, initial_state=[c2_fw, h2_fw])
        layer3_bw = keras.layers.LSTM(units=lstm_num_units, return_sequences=True, dropout=drop_rate,
                                      recurrent_dropout=drop_rate, go_backwards=True)(input_, initial_state=[c2_bw, h2_bw])
        layer3 = keras.layers.Concatenate(axis=2)([layer3_fw, layer3_bw])
        layer3_maxpooling = keras.layers.GlobalMaxPooling1D()(layer3)

        sen_encoder = keras.layers.Concatenate(axis=1)(
            [layer1_maxpooling, layer2_maxpooling, layer3_maxpooling])  

        return sen_encoder
