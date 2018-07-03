"""An implementation of CDSSM model."""
from matchzoo import engine
from keras.models import Model
from keras.layers import Dense, Input, Dot, Conv1D, Maximum


class CDSSMModel(engine.BaseModel):
    """
    Convolutional Deep structured semantic model.

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

    def build(self):
        """
        Build model structure.

        CDSSM use Siamese arthitecture.
        """
        # Left input and right input.
        dim_triletter = self._params['input_shapes'][0][0]
        num_tri_letters_le = self._params['input_shapes'][0][1]
        num_tri_letters_ri = self._params['input_shapes'][1][1]
        input_shape_le = (dim_triletter, 1)
        input_shape_ri = (dim_triletter, 1)
        # Create documen level network (Max-pooling and Dense).
        document_level_network_le = self._document_level_max_pooling(
            input_shape_le, num_tri_letters_le)
        document_level_network_ri = self._document_level_max_pooling(
            input_shape_ri, num_tri_letters_ri)
        # Create inputs.
        inputs_le = [Input(shape=(dim_triletter, 1))
                     for _
                     in range(num_tri_letters_le)]

        inputs_ri = [Input(shape=(dim_triletter, 1))
                     for _
                     in range(num_tri_letters_ri)]
        # Process left & right input.
        x = [document_level_network_le(inputs_le),
             document_level_network_ri(inputs_ri)]
        # Dot product with cosine similarity.
        x = Dot(axes=[1, 1],
                normalize=True)(x)
        # Make output layer and return Model.
        x_out = self._make_output_layer()(x)
        self._backend = Model(
            inputs=[inputs_le, inputs_ri],
            outputs=x_out)

    def _triletter_level_conv(self, input_shape: tuple) -> Model:
        """
        Apply convolutional operation towards to each tri-letter.

        :param input_shape: tuple of input shapes.

        :return: Keras `Model`, input 1 by n dimension tensor, output
                 300d tensor.
        """
        # TODO use sparse input in the future.
        # Input word hashing layer.
        input = Input(shape=input_shape)
        # Convolutional layer.
        x = Conv1D(filters=self._params['dim_conv'],
                   kernel_size=self._params['window_conv'],
                   strides=self._params['strides'],
                   padding=self._params['padding'],
                   kernel_initializer=self._params['w_initializer'],
                   bias_initializer=self._params['b_initializer'])(input)
        return Model(inputs=input, outputs=x)

    def _document_level_max_pooling(self, input_shape: tuple, num_tri_letters: int) -> Model:
        """
        Apply Max-pooling on all tri-letters.

        Take max at each dimension across all word-trigram features.
        Result in a (1 * 300) tensor representation, then further reduce
        the dimensionality use a Dense connected layer.

        :param: input_shape: a tuple indicate input shapes.

        :return: Keras Model, input list of tri-letters, output 128d tensor.
        """
        # Apply conv on each 90k dim tri-letter.
        # Result in a (num_triletters * 300) matrix.
        tri_letter_input_shape = (
            self._params['input_shapes'][0][0], 1)
        tri_letter_level_network = self._triletter_level_conv(
            input_shape=tri_letter_input_shape)
        # Dynamically created inputs.
        tri_letter_inputs = []
        for idx in range(num_tri_letters):
            tri_letter_inputs.append(
                Input(shape=tri_letter_input_shape))
        # Process tri-letter inputs
        x = [tri_letter_level_network(tri_letter_input)
             for tri_letter_input
             in tri_letter_inputs]
        # Take max at each dimension across all word-trigram features.
        # Result in a (1 * 300) tensor representation.
        # TODO Make sure element-wise maximum is correct.
        x = Maximum()(x)
        # Use Dense Layer to reduce 300 d to 128 d.
        x_out = Dense(self._params['dim_fan_out'],
                      activation=self._params['activation_dense'])(x)
        return Model(inputs=tri_letter_inputs,
                     outputs=x_out)
