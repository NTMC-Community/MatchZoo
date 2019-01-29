"""KNRM model."""
import keras
import keras.backend as K

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class KNRM(BaseModel):
    """
    KNRM model.

    Examples:
        >>> model = KNRM()
        >>> model.params['embedding_input_dim'] =  10000
        >>> model.params['embedding_output_dim'] =  10
        >>> model.params['embedding_trainable'] = True
        >>> model.params['kernel_num'] = 11
        >>> model.params['sigma'] = 0.1
        >>> model.params['exact_sigma'] = 0.001
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls):
        """Get default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(
            name='kernel_num',
            value=11,
            hyper_space=hyper_spaces.quniform(low=5, high=20),
            desc="The number of RBF kernels."
        ))
        params.add(Param(
            name='sigma',
            value=0.1,
            hyper_space=hyper_spaces.quniform(
                low=0.01, high=0.2, q=0.01),
            desc="The `sigma` defines the kernel width."
        ))
        params.add(Param(
            name='exact_sigma', value=0.001,
            desc="The `exact_sigma` denotes the `sigma` "
                 "for exact match."
        ))
        return params

    def build(self):
        """Build model."""
        query, doc = self._make_inputs()

        embedding = self._make_embedding_layer()
        q_embed = embedding(query)
        d_embed = embedding(doc)

        mm = keras.layers.Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])

        KM = []
        for i in range(self._params['kernel_num']):
            mu = 1. / (self._params['kernel_num'] - 1) + (2. * i) / (
                self._params['kernel_num'] - 1) - 1.0
            sigma = self._params['sigma']
            if mu > 1.0:
                sigma = self._params['exact_sigma']
                mu = 1.0
            mm_exp = self._kernel_layer(mu, sigma)(mm)
            mm_doc_sum = keras.layers.Lambda(
                lambda x: K.tf.reduce_sum(x, 2))(mm_exp)
            mm_log = keras.layers.Activation(K.tf.log1p)(mm_doc_sum)
            mm_sum = keras.layers.Lambda(
                lambda x: K.tf.reduce_sum(x, 1))(mm_log)
            KM.append(mm_sum)

        phi = keras.layers.Lambda(lambda x: K.tf.stack(x, 1))(KM)
        out = self._make_output_layer()(phi)
        self._backend = keras.Model(inputs=[query, doc], outputs=[out])

    @classmethod
    def _kernel_layer(cls, mu: float, sigma: float) -> keras.layers.Layer:
        """
        Gaussian kernel layer in KNRM.

        :param mu: Float, mean of the kernel.
        :param sigma: Float, sigma of the kernel.
        :return: `keras.layers.Layer`.
        """

        def kernel(x):
            return K.tf.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)

        return keras.layers.Activation(kernel)
