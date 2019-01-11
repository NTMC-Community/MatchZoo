"""An implementation of ArcI Model."""
import typing

import keras

from matchzoo import engine
from matchzoo import preprocessors


class ArcI(engine.BaseModel):
    """
    ArcI Model.

    Examples:
        >>> model = ArcI()
        >>> model.params['num_blocks'] = 1
        >>> model.params['left_filters'] = [32]
        >>> model.params['right_filters'] = [32]
        >>> model.params['left_kernel_sizes'] = [3]
        >>> model.params['right_kernel_sizes'] = [3]
        >>> model.params['left_pool_sizes'] = [2]
        >>> model.params['right_pool_sizes'] = [4]
        >>> model.params['conv_activation_func'] = 'relu'
        >>> model.params['mlp_num_layers'] = 1
        >>> model.params['mlp_num_units'] = 64
        >>> model.params['mlp_num_fan_out'] = 32
        >>> model.params['mlp_activation_func'] = 'relu'
        >>> model.params['dropout_rate'] = 0.5
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True,
            with_multi_layer_perceptron=True
        )
        params['optimizer'] = 'adam'
        params.add(engine.Param(name='num_blocks', value=1,
                                desc="Number of convolution blocks."))
        params.add(engine.Param(name='left_filters', value=[32],
                                desc="The filter size of each convolution "
                                     "blocks for the left input."))
        params.add(engine.Param(name='left_kernel_sizes', value=[3],
                                desc="The kernel size of each convolution "
                                     "blocks for the left input."))
        params.add(engine.Param(name='right_filters', value=[32],
                                desc="The filter size of each convolution "
                                     "blocks for the right input."))
        params.add(engine.Param(name='right_kernel_sizes', value=[3],
                                desc="The kernel size of each convolution "
                                     "blocks for the right input."))
        params.add(engine.Param(name='conv_activation_func', value='relu',
                                desc="The activation function in the "
                                     "convolution layer."))
        params.add(engine.Param(name='left_pool_sizes', value=[2],
                                desc="The pooling size of each convolution "
                                     "blocks for the left input."))
        params.add(engine.Param(name='right_pool_sizes', value=[2],
                                desc="The pooling size of each convolution "
                                     "blocks for the right input."))
        params.add(engine.Param(
            name='padding',
            value='same',
            hyper_space=engine.hyper_spaces.choice(
                ['same', 'valid', 'causal']),
            desc="The padding mode in the convolution layer. It should be one"
                 "of `same`, `valid`, and `causal`."
        ))
        params.add(engine.Param(
            'dropout_rate', 0.0,
            hyper_space=engine.hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        """
        Build model structure.

        ArcI use Siamese arthitecture.
        """
        input_left, input_right = self._make_inputs()

        embedding = self._make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        for i in range(self._params['num_blocks']):
            embed_left = self._conv_pool_block(
                embed_left,
                self._params['left_filters'][i],
                self._params['left_kernel_sizes'][i],
                self._params['padding'],
                self._params['conv_activation_func'],
                self._params['left_pool_sizes'][i]
            )
            embed_right = self._conv_pool_block(
                embed_right,
                self._params['right_filters'][i],
                self._params['right_kernel_sizes'][i],
                self._params['padding'],
                self._params['conv_activation_func'],
                self._params['right_pool_sizes'][i]
            )

        rep_left = keras.layers.Flatten()(embed_left)
        rep_right = keras.layers.Flatten()(embed_right)
        concat = keras.layers.Concatenate(axis=1)([rep_left, rep_right])
        dropout = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(concat)
        mlp = self._make_multi_layer_perceptron_layer()(dropout)

        inputs = [input_left, input_right]
        x_out = self._make_output_layer()(mlp)
        self._backend = keras.Model(inputs=inputs, outputs=x_out)

    def _conv_pool_block(
        self,
        input_: typing.Any,
        filters: int,
        kernel_size: int,
        padding: str,
        conv_activation_func: str,
        pool_size: int
    ) -> typing.Any:
        output = keras.layers.Conv1D(
            filters,
            kernel_size,
            padding=padding,
            activation=conv_activation_func
        )(input_)
        output = keras.layers.MaxPooling1D(pool_size=pool_size)(output)
        return output
