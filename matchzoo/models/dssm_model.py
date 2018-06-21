"""An implementation of DSSM, Deep Structured Semantic Model."""
from matchzoo import engine
from keras.models import Model
from keras.layers import Dense, Input, Dot


class DssmModel(engine.BaseModel):
    """
    Deep structured semantic model.

    Examples:
        >>> model = DssmModel()
        >>> model.params['lr'] = 0.2
        >>> model.guess_and_fill_missing_params()
        >>> if model.all_params_filled():
        ...     model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params.add(engine.Param('w_initializer', 'glorot_normal'))
        params.add(engine.Param('b_initializer', 'zeros'))
        # TODO GET TRI-LETTER DIMENSIONALITY FROM FIT-TRANSFORM AS INPUT SHAPE
        params.add(engine.Param('input_shapes', [(30000,), (30000,)]))
        params.add(engine.Param('dim_fan_out', 128))
        params.add(engine.Param('dim_hidden', 300))
        params.add(engine.Param('activation_hidden', 'tanh'))
        params.add(engine.Param('activation_prediction', 'softmax'))
        params.add(engine.Param('lr', 0.01))
        params.add(engine.Param('decay', 1e-6))
        params.add(engine.Param('batch_size', 1024))
        params.add(engine.Param('num_epochs', 20))
        params.add(engine.Param('train_test_split', 0.2))
        params.add(engine.Param('verbose', 1))
        params.add(engine.Param('shuffle', True))
        params.add(engine.Param('optimizer', 'sgd'))
        params.add(engine.Param('num_hidden_layers', 2))
        params.add(engine.Param('loss', 'categorical_crossentropy'))
        return params

    def build(self):
        """
        Build model structure.

        DSSM use Siamese arthitecture.
        """
        dim_triletter = self._params['input_shapes'][0][0]
        input_shape = (dim_triletter,)
        base_network = self._create_base_network(input_shape=input_shape)
        # Left input and right input.
        input_left = Input(shape=input_shape)
        input_right = Input(shape=input_shape)
        # Process left & right input.
        x = [base_network(input_left),
             base_network(input_right)]
        # Dot product with cosine similarity.
        x = Dot(axes=[1, 1],
                normalize=True)(x)
        x_out = Dense(2,
                      activation=self._params['activation_prediction'])(x)
        self._backend = Model(
            inputs=[input_left, input_right],
            outputs=x_out)

    def _create_base_network(self, input_shape: tuple) -> Model:
        """
        Build base to be shared.

        Word-hashing layer, hidden layer * 2,  out layer.

        :param: input_shape: shape of the input.

        :return: x: 128d vector(tensor) representation.
        """
        input = Input(shape=input_shape)
        x = Dense(self._params['dim_hidden'],
                  kernel_initializer=self._params['w_initializer'],
                  bias_initializer=self._params['b_initializer'])(input)
        for _ in range(0, self._params['num_hidden_layers']):
            x = Dense(self._params['dim_hidden'],
                      activation=self._params['activation_hidden'])(x)
        # Out layer, map tri-letters into 128d representation.
        x = Dense(self._params['dim_fan_out'],
                  activation=self._params['activation_hidden'])(x)
        return Model(input, x)
