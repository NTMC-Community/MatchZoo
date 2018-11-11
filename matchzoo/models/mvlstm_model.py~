"""An implementation of MvLstm Model."""
from matchzoo import engine
import typing
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dropout, \
    Concatenate, Flatten, LSTM, Dot, Reshape, Lambda, Flatten, Dense


class MvLstmModel(engine.BaseModel):
    """
    MvLstm Model.
    Examples:
        >>> model = MvLstmModel()
        >>> model.guess_and_fill_missing_params()
        >>> model.build()
    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params['optimizer'] = 'adam'
        params['input_shapes'] = [(32,), (32,)]
        params.add(engine.Param('trainable_embedding', False))
        params.add(engine.Param('embedding_dim', 300))
        params.add(engine.Param('vocab_size', 100))
        params.add(engine.Param('hidden_size', 32))
        params.add(engine.Param('top_k', 10))
        params.add(engine.Param('padding', 'same'))
        params.add(engine.Param('dropout_rate', 0.0))
        params.add(engine.Param('embedding_mat', None))
        params.add(engine.Param('embedding_random_scale', 0.0))
        return params

    @property
    def embedding_mat(self) -> np.ndarray:
        """Get pretrained embedding for MvLstm model."""
        # Check if provided embedding matrix
        if self._params['embedding_mat'] is None:
            s = self._params['embedding_random_scale']
            self._params['embedding_mat'] = \
                np.random.uniform(-s, s, (self._params['vocab_size'],
                                          self._params['embedding_dim']))
        return self._params['embedding_mat']

    @embedding_mat.setter
    def embedding_mat(self, embedding_mat: np.ndarray):
        """
        Set pretrained embedding for MvLstm model.
        :param embedding_mat: pretrained embedding in numpy format.
        """
        self._params['embedding_mat'] = embedding_mat
        self._params['vocab_size'], self._params['embedding_dim'] = \
            embedding_mat.shape

    def build(self):
        """
        Build model structure.
        MvLstm use Siamese arthitecture.
        """

        def _top_k(inputs: typing.Any, max_k: int) -> typing.Any:
            x = K.tf.nn.top_k(inputs, k=max_k, sorted=True)[0]
            return x

        # Left input and right input.
        input_left = Input(name='text_left',
                           shape=self._params['input_shapes'][0])
        input_right = Input(name='text_right',
                            shape=self._params['input_shapes'][1])
        # Process left & right input.
        embedding = Embedding(self._params['vocab_size'],
                              self._params['embedding_dim'],
                              weights=[self.embedding_mat],
                              trainable=self._params['trainable_embedding'])
        bi_lstm = Bidirectional(LSTM(self._params['hidden_size'],
                              return_sequences=True,
                              dropout=self._params['dropout_rate']))
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        rep_left = bi_lstm(embed_left)
        rep_right = bi_lstm(embed_right)
        cross = Dot(axes=[1, 1], normalize=False)([rep_left, rep_right])
        cross = Reshape((-1, ))(cross)

        mm_k = Lambda(_top_k, arguments={"max_k": self._params['top_k']})(cross)
        mm_k = Dropout(rate=self._params['dropout_rate'])(mm_k)

        x_out = self._make_output_layer()(mm_k)
        self._backend = Model(
            inputs=[input_left, input_right],
            outputs=x_out)
