"""An implementation of aNMM Model."""

import typing
import logging
import numpy as np

import keras
import keras.backend as K
from keras.activations import softmax
from  keras.initializers import RandomUniform
from keras.layers import *

from matchzoo import engine

logger = logging.getLogger(__name__)


def show_tensor_info(name: str, input: np.ndarray):
    """Show the tensor shapes."""
    logger.info(
        '[Layer]: %s\t[Shape]: %s\n' % (name, input.get_shape().as_list()))


class ANMMModel(engine.BaseModel):
    """
    ANMM Model.

    Examples:
        >>> model = ANMMModel()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(engine.Param('bin_num', 60))
        params.add(engine.Param('dropout_rate', 0.1))
        params.add(engine.Param('left_text_len', 10))
        params.add(engine.Param('num_layers', 2))
        params.add(engine.Param('hidden_sizes', [10, 10]))
        return params

    def build(self):
        """Build model structure."""
        # query is [batch_size, left_text_len]
        # doc is [batch_size, right_text_len, bin_num]
        query, doc = self._make_inputs()
        embedding = self._make_embedding_layer()
        initializer_fc = RandomUniform(minval=-0.1, maxval=0.1, seed=11)
        initializer_gate = RandomUniform(minval=-0.01, maxval=0.01, seed=11)

        q_embed = embedding(query)
        q_w = Dense(1, kernel_initializer=initializer_gate, use_bias=False)(q_embed)
        q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(self._params['left_text_len'],))(q_w)
        z = doc
        z = Dropout(rate=self._params['dropout_rate'])(z)
        for i in range(self._params['num_layers'] - 1):
            z = Dense(self._params['hidden_sizes'][i], kernel_initializer=initializer_fc)(z)
            z = Activation('tanh')(z)
        z = Dense(self._params['hidden_sizes'][self._params['num_layers'] - 1], kernel_initializer=initializer_fc)(z)
        z = Reshape((self._params['left_text_len'],))(z)
        q_w = Reshape((self._params['left_text_len'],))(q_w)
        x_out = Dot(axes=[1, 1])([z, q_w])
        self._backend = keras.Model(inputs=[query, doc], outputs=x_out)
