import keras

from engine.base_model import BaseModel
from engine.model_params import ModelParams


class DenseBaselineModel(BaseModel):
    """
    Examples:

        >>> from matchzoo.models import DenseBaselineModel
        >>> params = DenseBaselineModel.get_default_params()
        >>> params.num_dense_units = 1024
        >>> model = DenseBaselineModel(params)
        >>> [layer.name for layer in model.backend.layers]
        ['input_1', 'input_2', 'concatenate_1', 'dense_1', 'dense_2']
    """

    def __init__(self, params=None):
        super().__init__(params)

    @classmethod
    def get_default_params(cls):
        params = ModelParams()
        params.num_dense_units = 512
        return params

    def build(self):
        inputs = [keras.layers.Input((self._params.text_1_max_len,)),
                  keras.layers.Input((self._params.text_2_max_len,))]

        x = keras.layers.concatenate(inputs)
        x = keras.layers.Dense(self._params.num_dense_units, activation='relu')(x)
        x_out = keras.layers.Dense(1, activation='sigmoid')(x)

        self._backend = keras.models.Model(inputs=inputs, outputs=x_out)
