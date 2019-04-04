"""An implementation of ArcII Model."""
import typing

import keras

import matchzoo
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class ArcII(BaseModel):
    """
    ArcII Model.

    Examples:
        >>> model = ArcII()
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
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'
        opt_space = hyper_spaces.choice(['adam', 'rmsprop', 'adagrad'])
        params.get('optimizer').hyper_space = opt_space
        params.add(Param(name='num_blocks', value=1,
                         desc="Number of 2D convolution blocks."))
        params.add(Param(name='kernel_1d_count', value=32,
                         desc="Kernel count of 1D convolution layer."))
        params.add(Param(name='kernel_1d_size', value=3,
                         desc="Kernel size of 1D convolution layer."))
        params.add(Param(name='kernel_2d_count', value=[32],
                         desc="Kernel count of 2D convolution layer in"
                              "each block"))
        params.add(Param(name='kernel_2d_size', value=[[3, 3]],
                         desc="Kernel size of 2D convolution layer in"
                              " each block."))
        params.add(Param(name='activation', value='relu',
                         desc="Activation function."))
        params.add(Param(name='pool_2d_size', value=[[2, 2]],
                         desc="Size of pooling layer in each block."))
        params.add(Param(
            name='padding', value='same',
            hyper_space=hyper_spaces.choice(
                ['same', 'valid']),
            desc="The padding mode in the convolution layer. It should be one"
                 "of `same`, `valid`."
        ))
        params.add(Param(
            name='dropout_rate', value=0.0,
            hyper_space=hyper_spaces.quniform(low=0.0, high=0.8,
                                              q=0.01),
            desc="The dropout rate."
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
