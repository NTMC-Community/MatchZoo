"""An implementation of ArcI Model."""
from matchzoo import engine
import typing
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
        params.add(engine.Param('trainable_embedding', False))
        params.add(engine.Param('embedding_dim', 300))
        params.add(engine.Param('vocab_size', 100))
        params.add(engine.Param('num_blocks', 1))
        params.add(engine.Param('left_kernel_count', [32]))
        params.add(engine.Param('left_kernel_size', [3]))
        params.add(engine.Param('right_kernel_count', [32]))
        params.add(engine.Param('right_kernel_size', [3]))
        params.add(engine.Param('activation', 'relu'))
        params.add(engine.Param('left_pool_size', [16]))
        params.add(engine.Param('right_pool_size', [16]))
        params.add(engine.Param('padding', 'same'))
        params.add(engine.Param('dropout_rate', 0.0))
        params.add(engine.Param('embedding_mat',
                   np.random.uniform(-0.2, 0.2, (params['vocab_size'],
                                                 params['embedding_dim']))))
        return params

    def _conv_pool_block(self, input: typing.Any, kernel_count: int,
                         kernel_size: int, padding: str, activation: str,
                         pool_size: int) -> typing.Any:
        output = Conv1D(kernel_count,
                        kernel_size,
                        padding=padding,
                        activation=activation)(input)
        output = MaxPooling1D(pool_size=pool_size)(output)
        return output

    def build(self):
        """
        Build model structure.

        ArcI use Siamese arthitecture.
        """
        # Left input and right input.
        input_left = Input(name='text_left',
                           shape=self._params['input_shapes'][0])
        input_right = Input(name='text_right',
                            shape=self._params['input_shapes'][1])
        # Process left & right input.
        embedding = Embedding(self._params['vocab_size'],
                              self._params['embedding_dim'],
                              weights=[self._params['embedding_mat']],
                              trainable=self._params['trainable_embedding'])
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        for i in range(self._params['num_blocks']):
            embed_left = self._conv_pool_block(
                                     embed_left,
                                     self._params['left_kernel_count'][i],
                                     self._params['left_kernel_size'][i],
                                     self._params['padding'],
                                     self._params['activation'],
                                     self._params['left_pool_size'][i])
            embed_right = self._conv_pool_block(
                                     embed_right,
                                     self._params['right_kernel_count'][i],
                                     self._params['right_kernel_size'][i],
                                     self._params['padding'],
                                     self._params['activation'],
                                     self._params['right_pool_size'][i])

        embed_flat = Flatten()(Concatenate(axis=1)([embed_left, embed_right]))
        x = Dropout(rate=self._params['dropout_rate'])(embed_flat)

        x_out = self._make_output_layer()(x)
        self._backend = Model(
            inputs=[input_left, input_right],
            outputs=x_out)
