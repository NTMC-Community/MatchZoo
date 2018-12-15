"""An implementation of ArcI Model."""
import typing

import keras

from matchzoo import engine
from matchzoo import preprocessors


class ArcIModel(engine.BaseModel):
    """
    ArcI Model.

    Examples:
        >>> import matchzoo as mz
        >>> train_pack = mz.datasets.toy.load_data()
        >>> model = ArcIModel()
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
        params['embedding_output_dim'] = 300
        params.add(engine.Param('num_blocks', 1))
        params.add(engine.Param('left_kernel_count', [32]))
        params.add(engine.Param('left_kernel_sizes', [3]))
        params.add(engine.Param('right_kernel_count', [32]))
        params.add(engine.Param('right_kernel_sizes', [3]))
        params.add(engine.Param('activation', 'relu'))
        params.add(engine.Param('left_pool_sizes', [2]))
        params.add(engine.Param('right_pool_sizes', [2]))
        params.add(engine.Param(
            'padding',
            'same',
            hyper_space=engine.hyper_spaces.choice(['same', 'valid', 'causal'])
        ))
        params.add(engine.Param(
            'dropout_rate', 0.0,
            hyper_space=engine.hyper_spaces.quniform(low=0.0, high=0.8, q=0.01)
        ))
        return params

    @classmethod
    def get_default_preprocessor(cls):
        """:return: Instance of :class:`NaivePreprocessor`."""
        return preprocessors.NaivePreprocessor()

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
                self._params['left_kernel_count'][i],
                self._params['left_kernel_sizes'][i],
                self._params['padding'],
                self._params['activation'],
                self._params['left_pool_sizes'][i]
            )
            embed_right = self._conv_pool_block(
                embed_right,
                self._params['right_kernel_count'][i],
                self._params['right_kernel_sizes'][i],
                self._params['padding'],
                self._params['activation'],
                self._params['right_pool_sizes'][i]
            )

        concat = keras.layers.Concatenate(axis=1)([embed_left, embed_right])
        embed_flat = keras.layers.Flatten()(concat)
        x = keras.layers.Dropout(rate=self._params['dropout_rate'])(embed_flat)

        inputs = [input_left, input_right]
        x_out = self._make_output_layer()(x)
        self._backend = keras.Model(inputs=inputs, outputs=x_out)

    def _conv_pool_block(
        self,
        input_: typing.Any,
        kernel_count: int,
        kernel_size: int,
        padding: str,
        activation: str,
        pool_size: int
    ) -> typing.Any:
        output = keras.layers.Conv1D(
            kernel_count,
            kernel_size,
            padding=padding,
            activation=activation
        )(input_)
        output = keras.layers.MaxPooling1D(pool_size=pool_size)(output)
        return output
