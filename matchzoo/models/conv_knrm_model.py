"""ConvKNRM model."""

import keras
import keras.backend as K

from matchzoo import engine
from matchzoo import models


class ConvKNRMModel(models.KNRMModel):
    """
    ConvKNRM model.

    Examples:
        >>> model = ConvKNRMModel()
        >>> model.params['filters'] = 128
        >>> model.params['conv_activation_func'] = 'tanh'
        >>> model.params['max_ngram'] = 3
        >>> model.params['use_crossmatch'] = True
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    def get_default_params(cls):
        """Get default parameters."""
        params = super().get_default_params()
        params.add(engine.Param('filters', 128))
        params.add(engine.Param('conv_activation_func', 'relu'))
        params.add(engine.Param('max_ngram', 3))
        params.add(engine.Param('use_crossmatch', True))
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
                    print("non cross")
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
