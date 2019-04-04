"""An implementation of DRMM Model."""
import typing

import keras
import keras.backend as K

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class DRMM(BaseModel):
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
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True,
                                            with_multi_layer_perceptron=True)
        params.add(Param(name='mask_value', value=-1,
                         desc="The value to be masked from inputs."))
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
        query = keras.layers.Input(
            name='text_left',
            shape=self._params['input_shapes'][0]
        )
        match_hist = keras.layers.Input(
            name='match_histogram',
            shape=self._params['input_shapes'][1]
        )

        embedding = self._make_embedding_layer()
        # Process left input.
        # shape = [B, L, D]
        embed_query = embedding(query)
        # shape = [B, L]
        atten_mask = K.not_equal(query, self._params['mask_value'])
        # shape = [B, L]
        atten_mask = K.cast(atten_mask, K.floatx())
        # shape = [B, L, D]
        atten_mask = K.expand_dims(atten_mask, axis=2)
        # shape = [B, L, D]
        attention_probs = self.attention_layer(embed_query, atten_mask)

        # Process right input.
        # shape = [B, L, 1]
        dense_output = self._make_multi_layer_perceptron_layer()(match_hist)

        # shape = [B, 1, 1]
        dot_score = keras.layers.Dot(axes=[1, 1])(
            [attention_probs, dense_output])

        flatten_score = keras.layers.Flatten()(dot_score)

        x_out = self._make_output_layer()(flatten_score)
        self._backend = keras.Model(inputs=[query, match_hist], outputs=x_out)

    @classmethod
    def attention_layer(cls, attention_input: typing.Any,
                        attention_mask: typing.Any = None
                        ) -> keras.layers.Layer:
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
            dense_input = keras.layers.Lambda(
                lambda x: x + (1.0 - attention_mask) * -10000.0,
                name="attention_mask"
            )(dense_input)
        # shape = [B, L, 1]
        attention_probs = keras.layers.Lambda(
            lambda x: keras.layers.activations.softmax(x, axis=1),
            output_shape=lambda s: (s[0], s[1], s[2]),
            name="attention_probs"
        )(dense_input)
        return attention_probs
