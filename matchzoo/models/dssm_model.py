"""An implementation of DSSM, Deep Structured Semantic Model."""
import typing
from matchzoo import engine
from keras.models import Model
from keras.layers import Dense, Input, Dot, Activation


class DssmModel(engine.BaseModel):
    """
    Deep structured semantic model.

    Examples:
        >>> model = DssmModel()
        >>> model.params['lr'] = 0.2
        >>> model.guess_and_fill_missing_params()

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
        return params

    def build(self):
        """
        Build model structure.

        DSSM use pair-wise arthitecture.
        """
        dim_triletter = self._params['input_shapes'][0][0]
        x_in = [self._build_shared_model(dim_triletter),
                self._build_shared_model(dim_triletter)]
        # Dot product with cosine similarity.
        x = Dot(axes=1,
                normalize=True)(x_in)
        x_out = Activation(activation=self._params['activation_prediction'])(x)
        self._backend = Model(inputs=x_in, outputs=x_out)

    def _build_shared_model(self, dim_triletter: int) -> typing.Any:
        """
        Build common architecture share to adopt pair-wise input.

        Word-hashing layer, hidden layer * 2,  out layer.

        :returns:  128 dimension vector representation.
        """
        x_in = Input(shape=(dim_triletter,))
        # Initialize input layer.
        x = Dense(units=self._params['dim_hidden'],
                  input_shape=(dim_triletter,),
                  kernel_initializer=self._params['w_initializer'],
                  bias_initializer=self._params['b_initializer'])(x_in)
        # Add hidden layers.
        for _ in range(0, self._params['num_hidden_layers']):
            x = Dense(units=self._params['dim_hidden'],
                      input_shape=(self._params['dim_hidden'],),
                      activation=self._params['activation_hidden'])(x)
        # Out layer, map tri-letters into 128d representation.
        x_out = Dense(units=self._params['dim_fan_out'],
                      input_shape=(self._params['dim_hidden'],),
                      activation=self._params['activation_hidden'])(x)
        return x_out
