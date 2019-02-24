"""An implementation of DRMMTKS Model."""
import typing

import keras
import keras.backend as K

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class DRMMTKS(BaseModel):
    """
    DRMMTKS Model.

    Examples:
        >>> model = DRMMTKS()
        >>> model.params['embedding_input_dim'] = 10000
        >>> model.params['embedding_output_dim'] = 100
        >>> model.params['top_k'] = 20
        >>> model.params['mlp_num_layers'] = 1
        >>> model.params['mlp_num_units'] = 5
        >>> model.params['mlp_num_fan_out'] = 1
        >>> model.params['mlp_activation_func'] = 'tanh'
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True,
            with_multi_layer_perceptron=True
        )
        params.add(Param(name='mask_value', value=-1,
                         desc="The value to be masked from inputs."))
        params['input_shapes'] = [(5,), (300,)]
        params.add(Param(
            'top_k', value=10,
            hyper_space=hyper_spaces.quniform(low=2, high=100),
            desc="Size of top-k pooling layer."
        ))
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
        # shape = [B, L]
        atten_mask = K.not_equal(query, self._params['mask_value'])
        # shape = [B, L]
        atten_mask = K.cast(atten_mask, K.floatx())
        # shape = [B, L, 1]
        atten_mask = K.expand_dims(atten_mask, axis=2)
        # shape = [B, L, 1]
        attention_probs = self.attention_layer(embed_query, atten_mask)

        # Matching histogram of top-k
        # shape = [B, L, R]
        matching_matrix = keras.layers.Dot(axes=[2, 2], normalize=True)(
            [embed_query,
             embed_doc])
        # shape = [B, L, K]
        effective_top_k = min(self._params['top_k'],
                              self.params['input_shapes'][0][0],
                              self.params['input_shapes'][1][0])
        matching_topk = keras.layers.Lambda(
            lambda x: K.tf.nn.top_k(x, k=effective_top_k, sorted=True)[0]
        )(matching_matrix)

        # Process right input.
        # shape = [B, L, 1]
        dense_output = self._make_multi_layer_perceptron_layer()(matching_topk)

        # shape = [B, 1, 1]
        dot_score = keras.layers.Dot(axes=[1, 1])(
            [attention_probs, dense_output])

        flatten_score = keras.layers.Flatten()(dot_score)

        x_out = self._make_output_layer()(flatten_score)
        self._backend = keras.Model(inputs=[query, doc], outputs=x_out)

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
