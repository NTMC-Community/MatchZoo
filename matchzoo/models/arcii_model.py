"""An implementation of ArcII Model."""
import typing

import keras

import matchzoo
from matchzoo import engine


class ArcIIModel(engine.BaseModel):
    """
    ArcII Model.

    Examples:
        >>> model = ArcIIModel()
        >>> model.params['embedding_output_dim'] = 300
        >>> model.params['num_blocks'] = 2
        >>> model.params['kernel_1d_count'] = 32
        >>> model.params['kernel_1d_size'] = 3
        >>> model.params['kernel_2d_count'] = [16, 32]
        >>> model.params['kernel_2d_size'] = [[3, 3], [3, 3]]
        >>> model.params['pool_2d_size'] = [[2, 2], [2, 2]]
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'
        opt_space = engine.hyper_spaces.choice(['adam', 'rmsprop', 'adagrad'])
        params.get('optimizer').hyper_space = opt_space
        params.add(engine.Param('num_blocks', 1))
        params.add(engine.Param('kernel_1d_count', 32))
        params.add(engine.Param('kernel_1d_size', 3))
        params.add(engine.Param('kernel_2d_count', [32]))
        params.add(engine.Param('kernel_2d_size', [[3, 3]]))
        params.add(engine.Param('activation', 'relu'))
        params.add(engine.Param('pool_2d_size', [[2, 2]]))
        params.add(engine.Param(
            name='padding', value='same',
            hyper_space=engine.hyper_spaces.choice(['same', 'valid', 'causal'])
        ))
        params.add(engine.Param(
            name='dropout_rate', value=0.0,
            hyper_space=engine.hyper_spaces.quniform(low=0.0, high=0.8, q=0.01)
        ))
        return params

    def build(self):
        """
        Build model structure.

        ArcII has the desirable property of letting two sentences meet before
        their own high-level representations mature.
        """
        input_left, input_right = self._make_inputs()

        embedding = self._make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # Phrase level representations
        conv_1d_left = keras.layers.Conv1D(
            self._params['kernel_1d_count'],
            self._params['kernel_1d_size'],
            padding=self._params['padding']
        )(embed_left)
        conv_1d_right = keras.layers.Conv1D(
            self._params['kernel_1d_count'],
            self._params['kernel_1d_size'],
            padding=self._params['padding']
        )(embed_right)

        # Interaction
        matching_layer = matchzoo.layers.MatchingLayer(matching_type='plus')
        embed_cross = matching_layer([conv_1d_left, conv_1d_right])

        for i in range(self._params['num_blocks']):
            embed_cross = self._conv_pool_block(
                embed_cross,
                self._params['kernel_2d_count'][i],
                self._params['kernel_2d_size'][i],
                self._params['padding'],
                self._params['activation'],
                self._params['pool_2d_size'][i]
            )

        embed_flat = keras.layers.Flatten()(embed_cross)
        x = keras.layers.Dropout(rate=self._params['dropout_rate'])(embed_flat)

        inputs = [input_left, input_right]
        x_out = self._make_output_layer()(x)
        self._backend = keras.Model(inputs=inputs, outputs=x_out)

    @classmethod
    def _conv_pool_block(
        cls, x,
        kernel_count: int,
        kernel_size: int,
        padding: str,
        activation: str,
        pool_size: int
    ) -> typing.Any:
        output = keras.layers.Conv2D(kernel_count,
                                     kernel_size,
                                     padding=padding,
                                     activation=activation)(x)
        output = keras.layers.MaxPooling2D(pool_size=pool_size)(output)
        return output
