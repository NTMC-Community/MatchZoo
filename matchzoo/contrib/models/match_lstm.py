"""Match LSTM model."""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine import hyper_spaces


class MatchLSTM(BaseModel):
    """
    Match LSTM model.

    Examples:
        >>> model = MatchLSTM()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.params['embedding_input_dim'] = 10000
        >>> model.params['embedding_output_dim'] = 100
        >>> model.params['embedding_trainable'] = True
        >>> model.params['fc_num_units'] = 200
        >>> model.params['lstm_num_units'] = 256
        >>> model.params['dropout_rate'] = 0.5
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls):
        """Get default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(
            'lstm_num_units', 256,
            hyper_space=hyper_spaces.quniform(low=128, high=384, q=32),
            desc="The hidden size in the LSTM layer."
        ))
        params.add(Param(
            'fc_num_units', 200,
            hyper_space=hyper_spaces.quniform(
                low=100, high=300, q=20),
            desc="The hidden size in the full connection layer."
        ))
        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.9, q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        """Build model."""
        input_left, input_right = self._make_inputs()
        len_left = input_left.shape[1]
        len_right = input_right.shape[1]
        embedding = self._make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        lstm_left = layers.LSTM(self._params['lstm_num_units'],
                                return_sequences=True,
                                name='lstm_left')
        lstm_right = layers.LSTM(self._params['lstm_num_units'],
                                 return_sequences=True,
                                 name='lstm_right')
        encoded_left = lstm_left(embed_left)
        encoded_right = lstm_right(embed_right)

        tensor_left = tf.expand_dims(encoded_left, axis=2)
        tensor_right = tf.expand_dims(encoded_right, axis=1)
        tensor_left = K.repeat_elements(tensor_left, len_right, 2)
        tensor_right = K.repeat_elements(tensor_right, len_left, 1)
        tensor_merged = tf.concat([tensor_left, tensor_right], axis=-1)
        middle_output = layers.Dense(self._params['fc_num_units'],
                                     activation='tanh')(
            tensor_merged)
        attn_scores = layers.Dense(1)(middle_output)
        attn_scores = tf.squeeze(attn_scores, axis=3)
        exp_attn_scores = tf.math.exp(
            attn_scores - tf.reduce_max(attn_scores, axis=-1, keepdims=True))
        exp_sum = tf.reduce_sum(exp_attn_scores, axis=-1, keepdims=True)
        attention_weights = exp_attn_scores / exp_sum
        left_attn_vec = K.batch_dot(attention_weights, encoded_right)

        concat = layers.Concatenate(axis=1)(
            [left_attn_vec, encoded_right])
        lstm_merge = layers.LSTM(self._params['lstm_num_units'] * 2,
                                 return_sequences=False,
                                 name='lstm_merge')
        merged = lstm_merge(concat)
        dropout = layers.Dropout(
            rate=self._params['dropout_rate'])(merged)

        phi = layers.Dense(self._params['fc_num_units'],
                           activation='tanh')(dropout)
        inputs = [input_left, input_right]
        out = self._make_output_layer()(phi)
        self._backend = models.Model(inputs=inputs, outputs=[out])
