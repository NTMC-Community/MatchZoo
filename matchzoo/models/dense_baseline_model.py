"""A simple densely connected baseline model."""

import keras.layers

from matchzoo import engine


class DenseBaselineModel(engine.BaseModel):
    """
    A simple densely connected baseline model.

    Examples:
        >>> model = DenseBaselineModel()
        >>> model.params['input_shapes'] = [(30,), (30,)]
        >>> model.params['num_dense_units'] = 1024
        >>> model.guess_and_fill_missing_params()
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ModelParams:
        """:return: model default parameters."""
        params = super().get_default_params()
        params['num_dense_units'] = 512
        return params

    def build(self):
        """Model structure."""
        x_in = [keras.layers.Input(shape)
                for shape in self._params['input_shapes']]

        x = keras.layers.concatenate(x_in)
        x = keras.layers.Dense(
                self._params['num_dense_units'], activation='relu')(x)
        x_out = keras.layers.Dense(1, activation='sigmoid')(x)

        self._backend = keras.models.Model(inputs=x_in, outputs=x_out)
