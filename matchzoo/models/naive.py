"""Naive model with a simplest structure for testing purposes."""

import keras

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces


class Naive(BaseModel):
    """
    Naive model with a simplest structure for testing purposes.

    Bare minimum functioning model. The best choice to get things rolling.
    The worst choice to fit and evaluate performance.
    """

    @classmethod
    def get_default_params(cls):
        """Default parameters."""
        params = super().get_default_params()
        params.get('optimizer').hyper_space = \
            hyper_spaces.choice(['adam', 'adagrad', 'rmsprop'])
        return params

    def build(self):
        """Build."""
        x_in = self._make_inputs()
        x = keras.layers.concatenate(x_in)
        x_out = self._make_output_layer()(x)
        self._backend = keras.models.Model(inputs=x_in, outputs=x_out)
