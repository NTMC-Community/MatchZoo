"""An implementation of MatchPyramid Model."""
import typing

import keras

import matchzoo
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class MatchPyramid(BaseModel):
    """
    MatchPyramid Model.

    Examples:
        >>> model = MatchPyramid()
        >>> model.params['embedding_output_dim'] = 300
        >>> model.params['num_blocks'] = 2
        >>> model.params['kernel_count'] = [16, 32]
        >>> model.params['kernel_size'] = [[3, 3], [3, 3]]
        >>> model.params['dpool_size'] = [3, 10]
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(name='num_blocks', value=1,
                         desc="Number of convolution blocks."))
        params.add(Param(name='kernel_count', value=[32],
                         desc="The kernel count of the 2D convolution "
                              "of each block."))
        params.add(Param(name='kernel_size', value=[[3, 3]],
                         desc="The kernel size of the 2D convolution "
                              "of each block."))
        params.add(Param(name='activation', value='relu',
                         desc="The activation function."))
        params.add(Param(name='dpool_size', value=[3, 10],
                         desc="The max-pooling size of each block."))
        params.add(Param(
            name='padding', value='same',
            desc="The padding mode in the convolution layer."
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

        MatchPyramid text matching as image recognition.
        """
        input_left, input_right = self._make_inputs()
        input_dpool_index = keras.layers.Input(
            name='dpool_index',
            shape=[self._params['input_shapes'][0][0],
                   self._params['input_shapes'][1][0],
                   2],
            dtype='int32')

        embedding = self._make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # Interaction
        matching_layer = matchzoo.layers.MatchingLayer(matching_type='dot')
        embed_cross = matching_layer([embed_left, embed_right])

        for i in range(self._params['num_blocks']):
            embed_cross = self._conv_block(
                embed_cross,
                self._params['kernel_count'][i],
                self._params['kernel_size'][i],
                self._params['padding'],
                self._params['activation']
            )

        # Dynamic Pooling
        dpool_layer = matchzoo.layers.DynamicPoolingLayer(
            *self._params['dpool_size'])
        embed_pool = dpool_layer([embed_cross, input_dpool_index])

        embed_flat = keras.layers.Flatten()(embed_pool)
        x = keras.layers.Dropout(rate=self._params['dropout_rate'])(embed_flat)

        inputs = [input_left, input_right, input_dpool_index]
        x_out = self._make_output_layer()(x)
        self._backend = keras.Model(inputs=inputs, outputs=x_out)

    @classmethod
    def _conv_block(
        cls, x,
        kernel_count: int,
        kernel_size: int,
        padding: str,
        activation: str
    ) -> typing.Any:
        output = keras.layers.Conv2D(kernel_count,
                                     kernel_size,
                                     padding=padding,
                                     activation=activation)(x)
        return output
