"""An implementation of ArcI Model."""
from matchzoo import engine
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dropout, \
                         Concatenate, Flatten


class ArcIModel(engine.BaseModel):
    """
    ArcI Model.

    Examples:
        >>> model = ArcIModel()
        >>> model.guess_and_fill_missing_params()
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params['optimizer'] = 'adam'
        params['input_shapes'] = [(32,), (32,)]
        params['input_dtypes'] = [np.int32, np.int32]
        params.add(engine.Param('maxlen_left', 32))
        params.add(engine.Param('maxlen_right', 32))
        params.add(engine.Param('embed_finetune', False))
        params.add(engine.Param('embed_dim', 300))
        params.add(engine.Param('vocab_size', 100))
        params.add(engine.Param('left_kernel_count', 32))
        params.add(engine.Param('left_kernel_size', 3))
        params.add(engine.Param('right_kernel_count', 32))
        params.add(engine.Param('right_kernel_size', 3))
        params.add(engine.Param('q_pool_size', 16))
        params.add(engine.Param('d_pool_size', 16))
        params.add(engine.Param('dropout_rate', 0.0))
        params.add(engine.Param('embed',
                   np.random.uniform(-0.2, 0.2, (params['vocab_size'],
                                                 params['embed_dim']))))
        return params

    def build(self):
        """
        Build model structure.

        ArcI use Siamese arthitecture.
        """
        # Left input and right input.
        input_left = Input(name='id_left',
                           shape=(self._params['maxlen_left'], ))
        input_right = Input(name='id_right',
                            shape=(self._params['maxlen_right'], ))
        # Process left & right input.
        embedding = Embedding(self._params['vocab_size'],
                              self._params['embed_dim'],
                              weights=[self._params['embed']],
                              trainable=self._params['embed_finetune'])
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)
        conv_left = Conv1D(self._params['left_kernel_count'],
                           self._params['left_kernel_size'],
                           padding='same',
                           activation='relu')(embed_left)
        conv_right = Conv1D(self._params['right_kernel_count'],
                            self._params['right_kernel_size'],
                            padding='same',
                            activation='relu')(embed_right)
        pool_left = MaxPooling1D(
                        pool_size=self._params['q_pool_size'])(conv_left)
        pool_right = MaxPooling1D(
                        pool_size=self._params['d_pool_size'])(conv_right)
        pool_flat = Flatten()(Concatenate(axis=1)([pool_left, pool_right]))
        x = Dropout(rate=self._params['dropout_rate'])(pool_flat)

        x_out = self._make_output_layer()(x)
        self._backend = Model(
            inputs=[input_left, input_right],
            outputs=x_out)
