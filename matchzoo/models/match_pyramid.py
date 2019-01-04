"""An implementation of MatchPyramid Model."""
import typing

import keras

import matchzoo
from matchzoo import engine


class MatchPyramid(engine.BaseModel):
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
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'
        opt_space = engine.hyper_spaces.choice(['adam', 'rmsprop', 'adagrad'])
        params.get('optimizer').hyper_space = opt_space
        params.add(engine.Param('num_blocks', 1))
        params.add(engine.Param('kernel_count', [32]))
        params.add(engine.Param('kernel_size', [[3, 3]]))
        params.add(engine.Param('activation', 'relu'))
        params.add(engine.Param('dpool_size', [3, 10]))
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

        MatchPyramid text matching as image recognition.
        """
        input_left, input_right = self._make_inputs()
        input_dpool_index = keras.layers.Input(
            name='dpool_index',
            shape=[self._params['input_shapes'][0][0],
                   self._params['input_shapes'][1][0],
                   3],
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
