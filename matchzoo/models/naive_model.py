"""Naive model with a simplest structure for testing purposes."""

import keras

from matchzoo import engine


class NaiveModel(engine.BaseModel):
    """Naive model with a simplest structure for testing purposes."""

    def build(self):
        """Build."""
        x_in = self._make_inputs()
        x = keras.layers.concatenate(x_in)
        x_out = self._make_output_layer()(x)
        self._backend = keras.models.Model(inputs=x_in, outputs=x_out)
