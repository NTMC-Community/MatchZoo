"""An implementation of CDSSM (CLSM) model."""
import typing

from matchzoo import engine
from matchzoo import tasks
from matchzoo import TensorType
from matchzoo import preprocessors

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dot
from keras.layers import Dropout, GlobalMaxPool1D


class CDSSMModel(engine.BaseModel):
    """
    CDSSM Model implementation.

    Learning Semantic Representations Using Convolutional Neural Networks
    for Web Search. (2014a)
    A Latent Semantic Model with Convolutional-Pooling Structure for
    Information Retrieval. (2014b)

    Examples:
        >>> model = CDSSMModel()
        >>> model.params['optimizer'] = 'adam'
        >>> model.params['filters'] =  32
        >>> model.params['kernel_size'] = 3
        >>> model.params['mlp_hidden_units'] = [300, 300, 128]
        >>> model.params['conv_activation_func'] = 'relu'
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        # set :attr:`with_multi_layer_perceptron` to False to support
        # user-defined variable dense layer units
        params = super().get_default_params()
        params.add(engine.Param('filters', 32))
        params.add(engine.Param('kernel_size', 3))
        params.add(engine.Param('strides', 1))
        params.add(engine.Param('padding', 'same'))
        params.add(engine.Param('conv_activation_func', 'relu'))
        params.add(engine.Param('w_initializer', 'glorot_normal'))
        params.add(engine.Param('b_initializer', 'zeros'))
        params.add(engine.Param('dropout_rate', 0.3))
        params.add(engine.Param('mlp_activation_func', 'relu'))
        params.add(engine.Param('mlp_hidden_units', [64, 32]))
        return params

    def _create_base_network(self) -> typing.Callable:
        """
        Apply conv and maxpooling operation towards to each letter-ngram.

        The input shape is `fixed_text_length`*`number of letter-ngram`,
        as described in the paper, `n` is 3, `number of letter-trigram`
        is about 30,000 according to their observation.

        :return: Wrapped Keras `Layer` as CDSSM network, tensor in tensor out.
        """
        def _wrapper(x: TensorType):
            # Apply 1d convolutional on each word_ngram (lt).
            # Input shape: (batch_size, num_tri_letters, 90000)
            # Sequence of num_tri_letters vectors of 90000d vectors.
            x = Conv1D(filters=self._params['filters'],
                       kernel_size=self._params['kernel_size'],
                       strides=self._params['strides'],
                       padding=self._params['padding'],
                       activation=self._params['conv_activation_func'],
                       kernel_initializer=self._params['w_initializer'],
                       bias_initializer=self._params['b_initializer'])(x)
            # Apply max pooling by take max at each dimension across
            # all word_trigram features.
            x = Dropout(self._params['dropout_rate'])(x)
            x = GlobalMaxPool1D()(x)
            # Apply a none-linear transformation use a tanh layer.
            hidden_units = self._params['mlp_hidden_units']
            activation = self._params['mlp_activation_func']
            for unit in hidden_units:
                x = Dense(unit, activation=activation)(x)
            return x

        return _wrapper

    def build(self):
        """
        Build model structure.

        CDSSM use Siamese architecture.
        """
        input_shape = self._params['input_shapes'][0]
        base_network = self._create_base_network()
        # Left input and right input.
        input_left = Input(name='text_left', shape=input_shape)
        input_right = Input(name='text_right', shape=input_shape)
        # Process left & right input.
        x = [base_network(input_left),
             base_network(input_right)]
        # Dot product with cosine similarity.
        x = Dot(axes=[1, 1], normalize=True)(x)
        x_out = self._make_output_layer()(x)
        self._backend = Model(inputs=[input_left, input_right],
                              outputs=x_out)

    @classmethod
    def get_default_preprocessor(cls):
        """:return: Default preprocessor."""
        return preprocessors.CDSSMPreprocessor()

    def guess_and_fill_missing_params(self, verbose=1):
        """
        Guess and fill missing parameters in :attr:`params`.

        Use this method to automatically fill-in hyper parameters.
        This involves some guessing so the parameter it fills could be
        wrong. For example, the default task is `Ranking`, and if we do not
        set it to `Classification` manually for data packs prepared for
        classification, then the shape of the model output and the data will
        mismatch.

        :param verbose: Verbosity.
        """
        self._params.get('name').set_default(self.__class__.__name__, verbose)
        self._params.get('task').set_default(tasks.Ranking(), verbose)
        self._params.get('input_shapes').set_default([(10, 30), (10, 30)], verbose)
        self._params.get('optimizer').set_default('adam', verbose)
