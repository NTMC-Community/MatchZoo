"""ConvKNRM model."""

import keras
import keras.backend as K

from .knrm import KNRM
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class ConvKNRM(KNRM):
    """
    ConvKNRM model.

    Examples:
        >>> model = ConvKNRM()
        >>> model.params['embedding_input_dim'] = 10000
        >>> model.params['embedding_output_dim'] = 300
        >>> model.params['embedding_trainable'] = True
        >>> model.params['filters'] = 128
        >>> model.params['conv_activation_func'] = 'tanh'
        >>> model.params['max_ngram'] = 3
        >>> model.params['use_crossmatch'] = True
        >>> model.params['kernel_num'] = 11
        >>> model.params['sigma'] = 0.1
        >>> model.params['exact_sigma'] = 0.001
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    def get_default_params(cls):
        """Get default parameters."""
        params = super().get_default_params()
        params.add(Param(name='filters', value=128,
                         desc="The filter size in the convolution"
                              " layer."))
        params.add(Param(name='conv_activation_func', value='relu',
                         desc="The activation function in the "
                              "convolution layer."))
        params.add(Param(name='max_ngram', value=3,
                         desc="The maximum length of n-grams for the "
                              "convolution layer."))
        params.add(Param(name='use_crossmatch', value=True,
                         desc="Whether to match left n-grams and right "
                              "n-grams of different lengths"))
        return params

    def build(self):
        """Build model."""
        query, doc = self._make_inputs()

        embedding = self._make_embedding_layer()

        q_embed = embedding(query)
        d_embed = embedding(doc)

        q_convs = []
        d_convs = []
        for i in range(self._params['max_ngram']):
            c = keras.layers.Conv1D(
                self._params['filters'], i + 1,
                activation=self._params['conv_activation_func'],
                padding='same'
            )
            q_convs.append(c(q_embed))
            d_convs.append(c(d_embed))

        KM = []
        for qi in range(self._params['max_ngram']):
            for di in range(self._params['max_ngram']):
                # do not match n-gram with different length if use crossmatch
                if not self._params['use_crossmatch'] and qi != di:
                    continue
                q_ngram = q_convs[qi]
                d_ngram = d_convs[di]
                mm = keras.layers.Dot(axes=[2, 2],
                                      normalize=True)([q_ngram, d_ngram])

                for i in range(self._params['kernel_num']):
                    mu = 1. / (self._params['kernel_num'] - 1) + (2. * i) / (
                        self._params['kernel_num'] - 1) - 1.0
                    sigma = self._params['sigma']
                    if mu > 1.0:
                        sigma = self._params['exact_sigma']
                        mu = 1.0
                    mm_exp = self._kernel_layer(mu, sigma)(mm)
                    mm_doc_sum = keras.layers.Lambda(
                        lambda x: K.tf.reduce_sum(x, 2))(
                        mm_exp)
                    mm_log = keras.layers.Activation(K.tf.log1p)(mm_doc_sum)
                    mm_sum = keras.layers.Lambda(
                        lambda x: K.tf.reduce_sum(x, 1))(mm_log)
                    KM.append(mm_sum)

        phi = keras.layers.Lambda(lambda x: K.tf.stack(x, 1))(KM)
        out = self._make_output_layer()(phi)
        self._backend = keras.Model(inputs=[query, doc], outputs=[out])
