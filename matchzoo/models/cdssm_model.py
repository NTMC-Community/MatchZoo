"""An implementation of CDSSM (CLSM) model."""

from matchzoo import engine

from keras import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dot


class CDSSMModel(engine.BaseModel):
    """
    Convolutional deep structured semantic model.

    Examples:
        >>> model = CDSSMModel()
        >>> model.guess_and_fill_missing_params()
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params['optimizer'] = 'sgd'
        # TODO GET TRI-LETTER DIMENSIONALITY FROM FIT-TRANSFORM AS INPUT SHAPE
        # Dimension: (Contextual Sliding Window , NUM_TRI_LETTERS)
        params['input_shapes'] = [(90000, 10), (90000, 50)]
        params.add(engine.Param('w_initializer', 'glorot_normal'))
        params.add(engine.Param('b_initializer', 'zeros'))
        params.add(engine.Param('dim_fan_out', 128))
        params.add(engine.Param('dim_conv', 300))
        params.add(engine.Param('window_conv', 3))
        params.add(engine.Param('strides', 1))
        params.add(engine.Param('padding', 'same'))
        params.add(engine.Param('activation_dense', 'tanh'))
        return params

    def _create_base_network(self, input_shape: tuple) -> Model:
        """
        Apply convolutional operation towards to each tri-letter.

        The input shape is `num_word_ngrams` * (`window_conv`*`dim_triletter`),
        as described in the paper, `window_conv` is 3, according to their
        observation, `dim_triletter` is about 30,000, so each `word_ngram`
        is a 1 by 90,000 matrix.

        :param input_shape: tuple of input shapes.
        :return: Keras `Model`, input 1 by n dimension tensor, output
                 300d tensor.
        """
        # Input word hashing layer.
        input_ = Input(shape=input_shape)
        # Apply 1d convolutional on each word_ngram (lt).
        x = Conv1D(filters=self._params['dim_conv'],
                   kernel_size=self._params['window_conv'],
                   strides=self._params['strides'],
                   padding=self._params['padding'],
                   kernel_initializer=self._params['w_initializer'],
                   bias_initializer=self._params['b_initializer'])(input_)
        # Apply max pooling by take max at each dimension across
        # all word_trigram features.
        x = MaxPooling1D(pool_size=input_shape[0])(x)
        # Apply a none-linear transformation use tanh
        x_out = Dense(self._params['dim_fan_out'],
                      activation=self._params['activation_dense'])(x)
        return Model(inputs=input_, outputs=x_out)

    def build(self):
        """
        Build model structure.

        CDSSM use Siamese arthitecture.
        """
        input_shape_left = self._params['input_shapes'][0]
        input_shape_right = self._params['input_shapes'][1]
        base_network_left = self._create_base_network(
            input_shape=input_shape_left)
        base_network_right = self._create_base_network(
            input_shape=input_shape_right)
        # Left input and right input.
        input_left = Input(name='text_left', shape=input_shape_left)
        input_right = Input(name='text_right', shape=input_shape_right)
        # Process left & right input.
        x = [base_network_left(input_left),
             base_network_right(input_right)]
        # Dot product with cosine similarity.
        x = Dot(axes=[1, 1], normalize=True)(x)
        x_out = self._make_output_layer()(x)
        self._backend = Model(inputs=[input_left, input_right],
                              outputs=x_out)
