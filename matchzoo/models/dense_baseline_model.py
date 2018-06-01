import keras

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_model_params import BaseModelParams


class DenseBaselineModel(BaseModel):
    """
    Examples:

        >>> model = DenseBaselineModel()
        >>> model.params['num_dense_units'] = 1024
        >>> model.params['input_shapes'] = [(30,), (30,)]
        >>> model.build()
        >>> [layer.name for layer in model.backend.layers]
        ['input_1', 'input_2', 'concatenate_1', 'dense_1', 'dense_2']
    """

    @classmethod
    def get_default_params(cls):
        params = BaseModelParams()
        params['num_dense_units'] = 512
        return params

    def build(self):
        x_in = [keras.layers.Input(shape)
                for shape in self._params['input_shapes']]

        x = keras.layers.concatenate(x_in)
        x = keras.layers.Dense(
                self._params.num_dense_units, activation='relu')(x)
        x_out = keras.layers.Dense(1, activation='sigmoid')(x)

        self._backend = keras.models.Model(inputs=x_in, outputs=x_out)
