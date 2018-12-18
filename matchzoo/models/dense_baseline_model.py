"""A simple densely connected baseline model."""

import keras.layers

from matchzoo import engine


class DenseBaselineModel(engine.BaseModel):
    """
    A simple densely connected baseline model.

    Examples:
        >>> model = DenseBaselineModel()
        >>> model.params['mlp_num_units'] = 128
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()
        >>> model.compile()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_multi_layer_perceptron=True)
        params['mlp_num_units'] = 256
        params.get('mlp_num_units').hyper_space = \
            engine.hyper_spaces.quniform(16, 512)
        params.get('mlp_num_layers').hyper_space = \
            engine.hyper_spaces.quniform(1, 5)
        return params

    def build(self):
        """Model structure."""
        x_in = self._make_inputs()
        x = keras.layers.concatenate(x_in)
        x = self._make_multi_layer_perceptron_layer()(x)
        x_out = self._make_output_layer()(x)
        self._backend = keras.models.Model(inputs=x_in, outputs=x_out)
