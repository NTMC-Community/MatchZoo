"""An implementation of CDSSM, Deep Structured Semantic Model."""
from matchzoo import engine
from keras.models import Model
from keras.layers import Dense, Input, Dot, Conv1D
from keras import backend as K


class CDSSMModel(engine.BaseModel):
    """
    Deep structured semantic model.

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
        params['input_shapes'] = [(90000, 100), (90000, 500)]
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
        input_shape_le = self._params['input_shapes'][0]
        input_shape_ri = self._params['input_shapes'][1]
        input_le = Input(shape=input_shape_le)
        input_ri = Input(shape=input_shape_ri)
        # Process left & right input.
        x = [self._document_level_max_pooling(input_shape_le[1]),
             self._document_level_max_pooling(input_shape_ri[1])]
        # Dot product with cosine similarity.
        x = Dot(axes=[1, 1],
                normalize=True)(x)
        # Make output layer and return Model.
        x_out = self._make_output_layer()(x)
        self._backend = Model(
            inputs=[input_le, input_ri],
            outputs=x_out)

    def _triletter_level_conv(self) -> Model:
        """
        Apply convolutional operation towards to each tri-letter.

        :return: x: 300d vector(tensor) representation.
        """
        # TODO use sparse input in the future.
        # Input word hashing layer.
        input_shape = (self._params['input_shapes'][0][0],)
        input = Input(shape=input_shape)
        # Convolutional layer.
        x = Conv1D(filters=self._params['dim_conv'],
                   kernel_size=self._params['window_conv'],
                   strides=self._params['strides'],
                   padding=self._params['padding'],
                   kernel_initializer=self._params['w_initializer'],
                   bias_initializer=self._params['b_initializer'])(input)
        return Model(inputs=input, outputs=x)
    
    def _document_level_max_pooling(self, num_tri_letters: int) -> Model:
        """
        Apply Max-pooling on document (N Tri-letters).
        """
        tri_letter_inputs = []
        for idx in range(num_tri_letters):
            tri_letter_inputs.append(
                Input(shape=(self._params['dim_conv'],)))
        # Apply conv on each 90k dim tri-letter.
        # Result in a (num_triletters * 300) matrix.
        x =  K.map_fn(self._triletter_level_conv, tri_letter_inputs)
        # Take max at each dimension across all word-trigram features.
        # Result in a (1 * 300) tensor representation.
        x = K.max(axis=0, keepdims=True)(x)
        # Use Dense Layer to reduce 300 d to 128 d.
        x_out = Dense(self._params['dim_fan_out'],
                      activation=self._params['activation_dense'])(x)
        return Model(inputs=tri_letter_inputs,
                     outputs=x_out)
