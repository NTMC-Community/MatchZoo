"""An implementation of ESIM Model."""
import typing

import keras
from keras import backend as K
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class ESIM(BaseModel):
    """
    ESIM Model.

    Examples:
        >>> model = ESIM()
        >>> model.params['lstm_units'] = 64
        >>> model.params['mlp_num_layers'] = 0
        >>> model.params['mlp_num_units'] = 64
        >>> model.params['mlp_num_fan_out'] = 64
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

        embedding = self._make_embedding_layer()
        # Look up embedding matrix and get embed representation
        # shape = [B, L, D]
        embed_left = embedding(input_left)

        # shape = [B, R, D]
        embed_right = embedding(input_right)
        # Get the mask for calculating attention
        # shape = [B, L]
        mask_left = keras.layers.Lambda(get_mask)(input_left)
        # mask_left = keras.layers.Lambda(lambda x: K.tf.Print(x, [x]))(mask_left)

        # shape = [B, R]
        mask_right = keras.layers.Lambda(get_mask)(input_right)

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
        # encoded_left = keras.layers.Lambda(lambda x: K.tf.Print(x, [x]))(encoded_left)

        attended_left, attended_right = self._attention(encoded_left, mask_left, encoded_right, mask_right)
        # shape = [B, L, 8*H]
        enhanced_left = keras.layers.concatenate([encoded_left, attended_left,
                                                  keras.layers.subtract([encoded_left, attended_left]),
                                                  keras.layers.multiply([encoded_left, attended_left])])
        # shape = [B, R, 8*H]
        enhanced_right = keras.layers.concatenate([encoded_right, attended_right,
                                                   keras.layers.subtract([encoded_right, attended_right]),
                                                   keras.layers.multiply([encoded_right, attended_right])])
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
        avg_left_rep = keras.layers.Lambda(
            lambda x: K.sum(x * K.expand_dims(mask_left), axis=1) /
                      K.expand_dims(K.sum(mask_left, axis=1)))(fusion_left)
        avg_right_rep = keras.layers.Lambda(
            lambda x: K.sum(x * K.expand_dims(mask_right), axis=1) /
                      K.expand_dims(K.sum(mask_right, axis=1)))(fusion_right)
        max_left_rep = keras.layers.Lambda(
            lambda x: K.max(x + (1 - tile_tensor(mask_left, -1, [1, 1, 2 * hidden_size])) * -1e7,
                            axis=1))(fusion_left)
        max_right_rep = keras.layers.Lambda(
            lambda x: K.max(x + (1 - tile_tensor(mask_right, -1, [1, 1, 2 * hidden_size])) * -1e7,
                            axis=1))(fusion_right)
        # shape = [B, 8*H]
        mlp_input = keras.layers.concatenate([avg_left_rep, avg_right_rep, max_left_rep, max_right_rep])
        mlp_input = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(mlp_input)
        # Output layer
        mlp = self._make_multi_layer_perceptron_layer()(mlp_input)
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
        embed_cross = keras.layers.dot([encoded_left, encoded_right], axes=2)
        cross_shape = [self._params['input_shapes'][0][0], self._params['input_shapes'][1][0]]
        embed_cross = keras.layers.Reshape(cross_shape)(embed_cross)
        left_to_right_attn = self._masked_softmax(embed_cross, right_mask,
                                                  self._params['input_shapes'][0][0],
                                                  self._params['input_shapes'][1][0])

        right_to_left_attn = self._masked_softmax(keras.layers.Permute([2, 1])(embed_cross), left_mask,
                                                  self._params['input_shapes'][1][0],
                                                  self._params['input_shapes'][0][0])

        attended_left = self._weighted_sum(left_to_right_attn,
                                           encoded_right,
                                           left_mask)
        attended_right = self._weighted_sum(right_to_left_attn,
                                            encoded_left,
                                            right_mask)
        return attended_left, attended_right

    # def _masked_softmax(self, input, mask, att_len, base_len, epsilon=1e-8):
    #
    #     def tile_mask(x):
    #         res = K.tile(K.expand_dims(x, axis=1), [1, att_len, 1])
    #         return res
    #
    #     tiled_mask = keras.layers.Lambda(tile_mask)(mask)
    #     reshaped_mask = keras.layers.Lambda(lambda x: K.reshape(x, [-1, base_len]))(tiled_mask)
    #     reshaped_input = keras.layers.Lambda(lambda x: K.reshape(x, [-1, base_len]))(input)
    #     softmax_res = keras.layers.Softmax() \
    #         (keras.layers.multiply([reshaped_input, reshaped_mask]))
    #     softmax_res = keras.layers.multiply([softmax_res, reshaped_mask])
    #     softmax_res = keras.layers.Lambda(lambda x: x / K.sum(x, axis=-1, keepdims=True) + epsilon)(softmax_res)
    #     output = keras.layers.Lambda(lambda x: K.reshape(x, [-1, att_len, base_len]))(softmax_res)
    #     output = keras.layers.Lambda(lambda x: K.tf.Print(x, [x]))(output)
    #     return output

    def _masked_softmax(self, input, mask, att_len, base_len):
        inf = 1e8

        def tile_mask(x):
            res = inf * K.tile(K.expand_dims(1 - x, axis=1), [1, att_len, 1])
            return res

        tiled_mask = keras.layers.Lambda(tile_mask)(mask)
        # tiled_mask = keras.layers.Lambda(lambda x: K.tf.Print(x, [x]))(tiled_mask)

        # reshaped_mask = keras.layers.Lambda(lambda x: K.reshape(x, [-1, base_len]))(tiled_mask)
        # reshaped_input = keras.layers.Lambda(lambda x: K.reshape(x, [-1, base_len]))(input)
        # input = keras.layers.Lambda(lambda x: K.tf.Print(x, [x]))(input)
        logits = keras.layers.subtract([input, tiled_mask])
        softmax_res = keras.layers.Softmax()(logits)
        # output = keras.layers.Lambda(lambda x: K.reshape(x, [-1, att_len, base_len]))(softmax_res)
        softmax_res = keras.layers.Lambda(lambda x: K.tf.Print(x, [x]))(softmax_res)
        return softmax_res

    def _weighted_sum(self, weights, tensor, mask):
        weighted_sum = keras.layers.dot([weights, tensor], axes=[2, 1])
        expanded_mask = keras.layers.Lambda(K.expand_dims)(mask)
        return keras.layers.multiply([expanded_mask, weighted_sum])


def get_mask(x):
    boolean_mask = K.not_equal(x, 0)
    return K.cast(boolean_mask, K.dtype(x))


def tile_tensor(x, axis, time):
    res = K.tile(K.expand_dims(x, axis=axis), time)
    return res
