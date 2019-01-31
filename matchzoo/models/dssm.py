"""An implementation of DSSM, Deep Structured Semantic Model."""
from keras.models import Model
from keras.layers import Input, Dot

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.base_model import BaseModel
from matchzoo import preprocessors


class DSSM(BaseModel):
    """
    Deep structured semantic model.

    Examples:
        >>> model = DSSM()
        >>> model.params['mlp_num_layers'] = 3
        >>> model.params['mlp_num_units'] = 300
        >>> model.params['mlp_num_fan_out'] = 128
        >>> model.params['mlp_activation_func'] = 'relu'
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_multi_layer_perceptron=True)
        return params

    def build(self):
        """
        Build model structure.

        DSSM use Siamese arthitecture.
        """
        dim_triletter = self._params['input_shapes'][0][0]
        input_shape = (dim_triletter,)
        base_network = self._make_multi_layer_perceptron_layer()
        # Left input and right input.
        input_left = Input(name='text_left', shape=input_shape)
        input_right = Input(name='text_right', shape=input_shape)
        # Process left & right input.
        x = [base_network(input_left),
             base_network(input_right)]
        # Dot product with cosine similarity.
        x = Dot(axes=[1, 1], normalize=True)(x)
        x_out = self._make_output_layer()(x)
        self._backend = Model(
            inputs=[input_left, input_right],
            outputs=x_out)

    @classmethod
    def get_default_preprocessor(cls):
        """:return: Default preprocessor."""
        return preprocessors.DSSMPreprocessor()
