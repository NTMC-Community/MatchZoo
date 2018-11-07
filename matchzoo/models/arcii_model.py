"""An implementation of ArcII Model."""
from matchzoo import engine
from matchzoo import layers
import typing
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, Conv2D, MaxPooling2D, \
    Dropout, Flatten


class ArcIIModel(engine.BaseModel):
    """
    ArcII Model.

    Examples:
        >>> model = ArcIIModel()
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
        params.add(engine.Param('kernel_1d_count', 32))
        params.add(engine.Param('kernel_1d_size', 3))
        params.add(engine.Param('kernel_2d_count', [32]))
        params.add(engine.Param('kernel_2d_size', [[3, 3]]))
        params.add(engine.Param('activation', 'relu'))
        params.add(engine.Param('pool_2d_size', [[2, 2]]))
        params.add(engine.Param('padding', 'same'))
        params.add(engine.Param('dropout_rate', 0.0))
        params.add(engine.Param('embedding_mat', None))
        params.add(engine.Param('embedding_random_scale', 0.2))
        return params

    def _conv_pool_block(self, input: typing.Any, kernel_count: int,
                         kernel_size: int, padding: str, activation: str,
                         pool_size: int) -> typing.Any:
        output = Conv2D(kernel_count,
                        kernel_size,
                        padding=padding,
                        activation=activation)(input)
        output = MaxPooling2D(pool_size=pool_size)(output)
        return output

    @property
    def embedding_mat(self) -> np.ndarray:
        """Get pretrained embedding for ArcII model."""
        # Check if provided embedding matrix
        if self._params['embedding_mat'] is None:
            s = self._params['embedding_random_scale']
            self._params['embedding_mat'] = \
                np.random.uniform(-s, s, (self._params['vocab_size'],
                                          self._params['embedding_dim']))
        return self._params['embedding_mat']

    @embedding_mat.setter
    def embedding_mat(self, embedding_mat: np.ndarray):
        """
        Set pretrained embedding for ArcII model.

        :param embedding_mat: pretrained embedding in numpy format.
        """
        self._params['embedding_mat'] = embedding_mat
        self._params['vocab_size'], self._params['embedding_dim'] = \
            embedding_mat.shape

    def build(self):
        """
        Build model structure.

        ArcII has the desirable property of letting two sentences meet before
        their own high-level representations mature.
        """
        # Left input and right input.
        input_left = Input(name='text_left',
                           shape=self._params['input_shapes'][0])
        input_right = Input(name='text_right',
                            shape=self._params['input_shapes'][1])
        # Process left & right input.
        embedding = Embedding(self._params['vocab_size'],
                              self._params['embedding_dim'],
                              weights=[self.embedding_mat],
                              trainable=self._params['trainable_embedding'])
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # Phrase level representations
        conv_1d_left = Conv1D(self._params['kernel_1d_count'],
                              self._params['kernel_1d_size'],
                              padding=self._params['padding'])(embed_left)
        conv_1d_right = Conv1D(self._params['kernel_1d_count'],
                               self._params['kernel_1d_size'],
                               padding=self._params['padding'])(embed_right)

        # Interaction
        embed_cross = layers.MatchLayer(match_type='plus')([conv_1d_left,
                                                            conv_1d_right])

        for i in range(self._params['num_blocks']):
            embed_cross = self._conv_pool_block(
                embed_cross,
                self._params['kernel_2d_count'][i],
                self._params['kernel_2d_size'][i],
                self._params['padding'],
                self._params['activation'],
                self._params['pool_2d_size'][i]
            )

        embed_flat = Flatten()(embed_cross)
        x = Dropout(rate=self._params['dropout_rate'])(embed_flat)

        x_out = self._make_output_layer()(x)
        self._backend = Model(
            inputs=[input_left, input_right],
            outputs=x_out)
