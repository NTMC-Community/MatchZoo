"""An implementation of DRMM Model."""
import typing

import keras
import keras.backend as K

from matchzoo import engine


class DRMM(engine.BaseModel):
    """
    DRMM Model.

    Examples:
        >>> model = DRMM()
        >>> model.params['mlp_num_layers'] = 1
        >>> model.params['mlp_num_units'] = 5
        >>> model.params['mlp_num_fan_out'] = 1
        >>> model.params['mlp_activation_func'] = 'tanh'
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()
        >>> model.compile()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True,
                                            with_multi_layer_perceptron=True)
        params['optimizer'] = 'adam'
        params['input_shapes'] = [(5,), (5, 30,)]
        return params

    def build(self):
        """Build model structure."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   H = histogram size
        #   K = size of top-k

        # Left input and right input.
        # query: shape = [B, L]
        # doc: shape = [B, L, H]
        # Note here, the doc is the matching histogram between original query
        # and original document.
        query, doc = self._make_inputs()

        embedding = self._make_embedding_layer()
        # Process left input.
        # shape = [B, L, D]
        embed_query = embedding(query)
        # shape = [B, L, D]
        attention_probs = self.attention_layer(embed_query)

        # Process right input.
        # shape = [B, L, 1]
        dense_output = self._make_multi_layer_perceptron_layer()(doc)

        # shape = [B, 1, 1]
        dot_score = keras.layers.Dot(axes=[1, 1])(
            [attention_probs, dense_output])

        flatten_score = keras.layers.Flatten()(dot_score)

        x_out = self._make_output_layer()(flatten_score)
        self._backend = keras.Model(inputs=[query, doc], outputs=x_out)

    @classmethod
    def attention_layer(cls, attention_input: typing.Any,
                        attention_mask: typing.Any = None):
        """
        Performs attention on the input.

        :param attention_input: The input tensor for attention layer.
        :param attention_mask: A tensor to mask the invalid values.
        :return: The masked output tensor.
        """
        # shape = [B, L, 1]
        dense_input = keras.layers.Dense(1, use_bias=False)(attention_input)
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
            lambda x: keras.layers.activations.softmax(x, axis=1),
            output_shape=lambda s: (s[0], s[1], s[2])
        )(dense_input)
        return attention_probs
