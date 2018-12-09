"""An implementation of DRMMTKS Model."""
import typing
import logging

import keras
import keras.backend as K
from keras.activations import softmax

from matchzoo import engine

logger = logging.getLogger(__name__)


class DRMMTKSModel(engine.BaseModel):
    """
    DRMMTKS Model.

    Examples:
        >>> model = DRMMTKSModel()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'
        params['input_shapes'] = [(5,), (300,)]
        params.add(engine.Param(
            'top_k', value=10,
            hyper_space=engine.hyper_spaces.quniform(low=2, high=100)
        ))
        params.add(engine.Param('hidden_sizes', [5, 1]))
        return params

    def build(self):
        """Build model structure."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   K = size of top-k

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        query, doc = self._make_inputs()

        embedding = self._make_embedding_layer()
        # Process left input.
        # shape = [B, L, D]
        embed_query = embedding(query)
        # shape = [B, R, D]
        embed_doc = embedding(doc)
        # shape = [B, L, 1]
        attention_probs = self.attention_layer(embed_query)

        # Matching histogram of top-k
        # shape = [B, L, R]
        matching_matrix = keras.layers.Dot(axes=[2, 2], normalize=True)(
            [embed_query,
             embed_doc])
        # shape = [B, L, K]
        matching_topk = keras.layers.Lambda(
            lambda x: K.tf.nn.top_k(x, k=self._params['top_k'], sorted=True)[0]
        )(matching_matrix)

        # Process right input.
        # shape = [B, L, 1]
        dense_output = self.multi_layer_perceptron(matching_topk)

        # shape = [B, 1, 1]
        dot_score = keras.layers.Dot(axes=[1, 1])(
            [attention_probs, dense_output])

        flatten_score = keras.layers.Flatten()(dot_score)

        x_out = self._make_output_layer()(flatten_score)
        self._backend = keras.Model(inputs=[query, doc], outputs=x_out)

    def attention_layer(self, input: typing.Any,
                        attention_mask: typing.Any = None):
        """
        Performs attention on the input.

        :param input: The input tensor for attention layer.
        :param attention_mask: A tensor to mask the invalid values.
        :return: The output tensor.
        """
        # shape = [B, L, 1]
        dense_input = keras.layers.Dense(1, use_bias=False)(input)
        if attention_mask is not None:
            # Since attention_mask is 1.0 for positions we want to attend and
            # 0.0 for masked positions, this operation will create a tensor
            # which is 0.0 for positions we want to attend and -10000.0 for
            # masked positions.

            # shape = [B, L, 1]
            adder = (1.0 - K.tf.cast(attention_mask, K.tf.float32)) * -1000.0
            # shape = [B, L, 1]
            dense_input += adder
        # shape = [B, L, 1]
        attention_probs = keras.layers.Lambda(
            lambda x: softmax(x, axis=1),
            output_shape=self._params['input_shapes'][0]
        )(dense_input)
        return attention_probs

    def multi_layer_perceptron(self, input: typing.Any):
        """
        Multiple Layer Perceptron.

        :param input: The input tensor for multi-layer perceptron.
        :return: The output tensor.
        """
        # shape = [B, L, H]
        out = input
        for idx, hidden_size in enumerate(self._params['hidden_sizes']):
            out = keras.layers.Dense(hidden_size)(out)
            out = keras.layers.Activation('tanh')(out)
        # shape = [B, L, 1]
        return out
