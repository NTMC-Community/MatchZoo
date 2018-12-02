"""An implementation of CDSSM (CLSM) model."""

from matchzoo import engine

from keras import Model
from keras.layers import Input, Conv1D, GlobalMaxPool1D, Dot, Dense


class CDSSMModel(engine.BaseModel):
    """
    CDSSM Model implementation.

    A Latent Semantic Model with Convolutional-Pooling Structure for
    Information Retrieval.
    Learning Semantic Representations Using Convolutional Neural Networks
    for Web Search.

    Examples:
        >>> model = CDSSMModel()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params['optimizer'] = 'adam'
        params['input_shapes'] = [(10, 900), (10, 900)]
        params.add(engine.Param('w_initializer', 'glorot_normal'))
        params.add(engine.Param('b_initializer', 'zeros'))
        params.add(engine.Param('dim_fan_out', 128))
        params.add(engine.Param('dim_hidden', 300))
        params.add(engine.Param('contextual_window', 3))
        params.add(engine.Param('strides', 1))
        params.add(engine.Param('padding', 'same'))
        params.add(engine.Param('activation_hidden', 'tanh'))
        params.add(engine.Param('num_hidden_layers', 1))
        return params

    def _create_base_network(self, input_shape: tuple) -> Model:
        """
        Apply conv and maxpooling operation towards to each tri-letter.

        The input shape is `num_word_ngrams`*(`contextual_window`*
        `dim_triletter`), as described in the paper, `contextual_window`
        is 3, according to their observation, `dim_triletter` is about
        30,000, so each `word_ngram` is a 1 by 90,000 matrix.

        :param input_shape: tuple of input shapes.
        :return: Keras `Model`, input 1 by n dimension tensor, output
                 128d tensor.
        """
        # Input word hashing layer.
        x_in = Input(shape=input_shape)
        # Apply 1d convolutional on each word_ngram (lt).
        # Input shape: (batch_size, num_tri_letters, 90000)
        # Sequence of num_tri_letters vectors of 90000d vectors.
        x = Conv1D(filters=self._params['dim_hidden'],
                   kernel_size=self._params['contextual_window'],
                   strides=self._params['strides'],
                   padding=self._params['padding'],
                   activation=self._params['activation_hidden'],
                   kernel_initializer=self._params['w_initializer'],
                   bias_initializer=self._params['b_initializer'])(x_in)
        # Apply max pooling by take max at each dimension across
        # all word_trigram features.
        x = GlobalMaxPool1D()(x)
        # Apply a none-linear transformation use a tanh layer.
        for _ in range(0, self._params['num_hidden_layers']):
            x = Dense(self._params['dim_fan_out'],
                      activation=self._params['activation_hidden'])(x)
        return Model(inputs=x_in, outputs=x)

    def build(self):
        """
        Build model structure.

        CDSSM use Siamese arthitecture.
        """
        input_shape = self._params['input_shapes'][0]
        base_network = self._create_base_network(
            input_shape=input_shape)
        # Left input and right input.
        input_left = Input(name='text_left', shape=input_shape)
        input_right = Input(name='text_right', shape=input_shape)
        # Process left & right input.
        x = [base_network(input_left),
             base_network(input_right)]
        # Dot product with cosine similarity.
        x = Dot(axes=[1, 1], normalize=True)(x)
        x_out = self._make_output_layer()(x)
        self._backend = Model(inputs=[input_left, input_right],
                              outputs=x_out)
