"""An implementation of ESIM Model."""
import typing

import keras
from keras import backend as K
import matchzoo
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class ESIM(BaseModel):
    """
    ESIM Model.

    Examples:
        >>> model = ESIM()
        >>> model.params['lstm_units'] = 256
        >>> model.params['mlp_num_layers'] = 0
        >>> model.params['mlp_num_units'] = 256
        >>> model.params['mlp_num_fan_out'] = 256
        >>> model.params['mlp_activation_func'] = 'tanh'
        >>> model.params['dropout_rate'] = 0.5
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
        params['optimizer'] = 'adam'
        params.add(Param(name='num_layers', value=1,
                         desc="Number of LSTM layers."))
        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        params.add(Param(
            name='lstm_units',
            value=256,
            desc="The hidden dimension in LSTM cell."
        ))
        return params

    def build(self):
        """
        Build model structure.

        """
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   H = LSTM hidden size

        # Left input and right input
        # shape = [B, L]
        # shape = [B, R]
        input_left, input_right = self._make_inputs()
        hidden_size = self._params['lstm_units']

        def tile_tensor(x):
            res = K.tile(K.expand_dims(x), [1, 1, 2 * hidden_size])
            return res

        embedding = self._make_embedding_layer()
        # Look up embedding matrix and get embed representation
        # shape = [B, L, D]
        embed_left = embedding(input_left)
        # shape = [B, R, D]
        embed_right = embedding(input_right)
        # Get the mask for calculating attention
        # shape = [B, L]
        left_mask = keras.layers.Lambda(get_mask)(input_left)
        # left_mask = keras.layers.Lambda(lambda x: K.tf.Print(x, [x]))(left_mask)

        # shape = [B, R]
        right_mask = keras.layers.Lambda(get_mask)(input_right)

        # Add dropout layer
        embed_left = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(embed_left)
        embed_right = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(embed_right)

        encoder_layer = keras.layers.Bidirectional(keras.layers.LSTM(
            hidden_size,
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))
        # Encode the two text
        # shape = [B, L, 2*H]
        encoded_left = encoder_layer(embed_left)
        # shape = [B, R, 2*H]
        encoded_right = encoder_layer(embed_right)
        # Make interaction and get the interactive representation
        # shape = [B, L, 2*H]
        # shape = [B, R, 2*H]
        attended_left, attended_right = self._attention(encoded_left, left_mask, encoded_right, right_mask)
        # shape = [B, L, 8*H]
        enhanced_left = keras.layers.Concatenate()([encoded_left, attended_left,
                                                    keras.layers.Subtract()([encoded_left, attended_left]),
                                                    keras.layers.Multiply()([encoded_left, attended_left])])
        # shape = [B, R, 8*H]
        enhanced_right = keras.layers.Concatenate()([encoded_right, attended_right,
                                                     keras.layers.Subtract()([encoded_right, attended_right]),
                                                     keras.layers.Multiply()([encoded_right, attended_right])])
        # Project to the H size
        projection = keras.layers.Dense(hidden_size, activation=keras.layers.ReLU())
        # shape = [B, L, H]
        projected_left = projection(enhanced_left)
        # shape = [B, R, H]
        projected_right = projection(enhanced_right)
        projected_left = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(projected_left)
        projected_right = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(projected_right)
        # Fusion the enhanced features by another BiLSTM
        fusion_layer = keras.layers.Bidirectional(keras.layers.LSTM(
            hidden_size,
            return_sequences=True,
            dropout=self._params['dropout_rate']
        ))
        # shape = [B, L, 2*H]
        fusion_left = fusion_layer(projected_left)
        # shape = [B, R, 2*H]
        fusion_right = fusion_layer(projected_right)
        # Do global avg and max pooling and get a fixed-length representation vector
        avg_left_rep = keras.layers.Lambda(lambda x: K.sum(x * expand_tensor(left_mask), axis=1) /
                                                     expand_tensor(K.sum(left_mask, axis=1)))(fusion_left)
        avg_right_rep = keras.layers.Lambda(lambda x: K.sum(x * expand_tensor(right_mask), axis=1) /
                                                      expand_tensor(K.sum(right_mask, axis=1)))(fusion_right)
        max_left_rep = keras.layers.Lambda(lambda x: K.max(x + (1 - tile_tensor(left_mask)) * -1e7, axis=1))(
            fusion_left)
        max_right_rep = keras.layers.Lambda(lambda x: K.max(x + (1 - tile_tensor(right_mask)) * -1e7, axis=1))(
            fusion_right)
        # shape = [B, 8*H]
        cls_input = keras.layers.Concatenate()([avg_left_rep, avg_right_rep, max_left_rep, max_right_rep])
        cls_input = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(cls_input)
        # Output layer
        mlp = self._make_multi_layer_perceptron_layer()(cls_input)
        mlp = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp)
        inputs = [input_left, input_right]
        # mlp = keras.layers.Lambda(lambda x: K.tf.Print(x, [x]))(mlp)
        x_out = self._make_output_layer()(mlp)
        self._backend = keras.Model(inputs=inputs, outputs=x_out)

    def _attention(self, encoded_left,
                   left_mask,
                   encoded_right,
                   right_mask):
        # Interaction
        embed_cross = keras.layers.Dot(axes=2)([encoded_left, encoded_right])
        cross_shape = [self._params['input_shapes'][0][0], self._params['input_shapes'][1][0]]
        embed_cross = keras.layers.Reshape(cross_shape)(embed_cross)
        left2right_attn = self._masked_softmax(embed_cross, left_mask,
                                               self._params['input_shapes'][0][0],
                                               self._params['input_shapes'][1][0])
        # left2right_attn = keras.layers.Lambda(lambda x: K.tf.Print(x, [x]))(left2right_attn)

        right2left_attn = self._masked_softmax(keras.layers.Permute([2, 1])(embed_cross), right_mask,
                                               self._params['input_shapes'][1][0],
                                               self._params['input_shapes'][0][0])

        attended_left = self._weighted_sum(left2right_attn,
                                           encoded_right,
                                           left_mask)
        attended_right = self._weighted_sum(right2left_attn,
                                            encoded_left,
                                            right_mask)
        return attended_left, attended_right

    def _masked_softmax(self, input, mask, att_len, base_len):
        inf = 1e8

        def tile_mask(x):
            res = inf * K.tile(K.expand_dims(1 - x, axis=1), [1, base_len, 1])
            return res

        # def fill_mask(x):
        #     return inf * x

        tiled_mask = keras.layers.Lambda(tile_mask)(mask)
        flattened_mask = keras.layers.Flatten()(tiled_mask)
        flattened_input = keras.layers.Flatten()(input)
        softmax_res = keras.layers.Softmax() \
            (keras.layers.Subtract()([flattened_input, flattened_mask]))
        output = keras.layers.Reshape([att_len, base_len])(softmax_res)
        return output

    def _weighted_sum(self, weights, tensor, mask):
        weighted_sum = keras.layers.Dot(axes=[2, 1])([weights, tensor])
        expanded_mask = keras.layers.Lambda(expand_tensor)(mask)
        return keras.layers.Multiply()([expanded_mask, weighted_sum])


def get_mask(x):
    boolean_mask = K.not_equal(x, 0)
    return K.cast(boolean_mask, K.dtype(x))


def expand_tensor(x):
    return K.expand_dims(x)


def sum_tensor(x):
    return K.sum(x, axis=1)
