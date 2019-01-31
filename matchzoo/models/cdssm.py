"""An implementation of CDSSM (CLSM) model."""
import typing

import keras
from keras.models import Model

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo import preprocessors
from matchzoo.utils import TensorType


class CDSSM(BaseModel):
    """
    CDSSM Model implementation.

    Learning Semantic Representations Using Convolutional Neural Networks
    for Web Search. (2014a)
    A Latent Semantic Model with Convolutional-Pooling Structure for
    Information Retrieval. (2014b)

    Examples:
        >>> model = CDSSM()
        >>> model.params['optimizer'] = 'adam'
        >>> model.params['filters'] =  32
        >>> model.params['kernel_size'] = 3
        >>> model.params['conv_activation_func'] = 'relu'
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        # set :attr:`with_multi_layer_perceptron` to False to support
        # user-defined variable dense layer units
        params = super().get_default_params(with_multi_layer_perceptron=True)
        params.add(Param(name='filters', value=32,
                         desc="Number of filters in the 1D convolution "
                              "layer."))
        params.add(Param(name='kernel_size', value=3,
                         desc="Number of kernel size in the 1D "
                              "convolution layer."))
        params.add(Param(name='strides', value=1,
                         desc="Strides in the 1D convolution layer."))
        params.add(Param(name='padding', value='same',
                         desc="The padding mode in the convolution "
                              "layer. It should be one of `same`, "
                              "`valid`, ""and `causal`."))
        params.add(Param(name='conv_activation_func', value='relu',
                         desc="Activation function in the convolution"
                              " layer."))
        params.add(Param(name='w_initializer', value='glorot_normal'))
        params.add(Param(name='b_initializer', value='zeros'))
        params.add(Param(name='dropout_rate', value=0.3,
                         desc="The dropout rate."))
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
            x = keras.layers.Conv1D(
                filters=self._params['filters'],
                kernel_size=self._params['kernel_size'],
                strides=self._params['strides'],
                padding=self._params['padding'],
                activation=self._params['conv_activation_func'],
                kernel_initializer=self._params['w_initializer'],
                bias_initializer=self._params['b_initializer'])(x)
            # Apply max pooling by take max at each dimension across
            # all word_trigram features.
            x = keras.layers.Dropout(self._params['dropout_rate'])(x)
            x = keras.layers.GlobalMaxPool1D()(x)
            # Apply a none-linear transformation use a tanh layer.
            x = self._make_multi_layer_perceptron_layer()(x)
            return x

        return _wrapper

    def build(self):
        """
        Build model structure.

        CDSSM use Siamese architecture.
        """
        base_network = self._create_base_network()
        # Left input and right input.
        input_left, input_right = self._make_inputs()
        # Process left & right input.
        x = [base_network(input_left),
             base_network(input_right)]
        # Dot product with cosine similarity.
        x = keras.layers.Dot(axes=[1, 1], normalize=True)(x)
        x_out = self._make_output_layer()(x)
        self._backend = Model(inputs=[input_left, input_right],
                              outputs=x_out)

    @classmethod
    def get_default_preprocessor(cls):
        """:return: Default preprocessor."""
        return preprocessors.CDSSMPreprocessor()

    def guess_and_fill_missing_params(self, verbose: int = 1):
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
        self._params.get('input_shapes').set_default([(10, 30),
                                                      (10, 30)], verbose)
        super().guess_and_fill_missing_params(verbose)
