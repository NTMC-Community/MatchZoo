"""Naive model with a simplest structure for testing purposes."""

import numpy as np
from keras.models import Model
from keras.layers import Input, Bidirectional, LSTM, Dense, Concatenate


from matchzoo import engine
from matchzoo.layers import MultiPerspectiveLayer


class BimpmModel(engine.BaseModel):
    """Naive model with a simplest structure for testing purposes."""

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params['optimizer'] = 'adam'
        params['input_shapes'] = [(32,), (32,)]
        params.add(engine.Param('dim_word_embedding', 300))
        params.add(engine.Param('dim_char_embedding', 50))
        params.add(engine.Param('word_embedding_mat', None))
        params.add(engine.Param('char_embedding_mat', None))
        params.add(engine.Param('embedding_random_scale', 0.2))
        params.add(engine.Param('activation_embedding', 'softmax'))
        params.add(engine.Param('vocab_size', 100))
        params.add(engine.Param('w_initializer', 'glorot_uniform'))
        params.add(engine.Param('b_initializer', 'zeros'))
        params.add(engine.Param('activation_hidden', 'linear'))
        params.add(engine.Param('dim_hidden', 128))
        params.add(engine.Param('perspective', {'full': True,
                                                'max-pooling': True,
                                                'attentive': True,
                                                'max-attentive': True}))
        return params

    @property
    def word_embedding_mat(self) -> np.ndarray:
        """Get pretrained embedding for ArcI model."""
        # Check if provided embedding matrix
        if self._params['word_embedding_mat'] is None:
            raise TypeError('No pre-trained word embeddings provided.')

    @word_embedding_mat.setter
    def word_embedding_mat(self, embedding_mat: np.ndarray):
        """
        Set pretrained embedding for ArcI model.

        :param embedding_mat: pretrained embedding in numpy format.
        """
        self._params['word_embedding_mat'] = embedding_mat
        self._params['vocab_size'], self._params['dim_word_embedding'] = \
            embedding_mat.shape

    @property
    def char_embedding_mat(self) -> np.ndarray:
        """Initialize character level embedding."""
        s = self._params['embedding_random_scale']
        self._params['char_embedding_mat'] = \
            np.random.uniform(-s, s, (self._params['vocab_size'],
                                      self._params['dim_char_embedding']))
        return self._params['char_embedding_mat']

    def char_embedding(self, char_input_shape, char_vocab_size):
        """Create a character level embedding model."""
        input_char = Input(shape=char_input_shape)
        embed_char = LSTM(self._params['dim_char_embedding'],
                          kernel_initializer=self._params['w_initializer'],
                          bias_initializer=self._params['b_initializer'])(
                              input_char)
        embed_char = Dense(char_vocab_size,
                           activation=self._params['activation_embedding'])(
                               embed_char)
        return Model(input_char, embed_char)

    def build(self):
        """Build."""
        input_shape_lt = self._params['input_shapes'][0]
        input_shape_rt = self._params['input_shapes'][1]
        input_lt = Input(shape=input_shape_lt)
        input_rt = Input(shape=input_shape_rt)
        # Word representation layer.
        # TODO Concanate word level embedding and character level embedding.
        # Context represntation layer.
        x_lt = Bidirectional(
            LSTM(input_shape_lt[0],
                 return_sequences=True,
                 return_state=True,
                 kernel_initializer=self._params['w_initializer'],
                 bias_initializer=self._params['b_initializer']),
            merge_mode=None)(input_lt)
        x_rt = Bidirectional(
            LSTM(input_shape_rt[0],
                 return_sequences=True,
                 return_state=True,
                 kernel_initializer=self._params['w_initializer'],
                 bias_initializer=self._params['b_initializer']),
            merge_mode=None)(input_rt)
        # Multiperspective Matching layer.
        # Output is two sequence of vectors.
        # TODO Finalize MultiPerspectiveMatching
        x_lt = MultiPerspectiveLayer(
            dim_output=(MultiPerspectiveLayer.num_perspective,
                        self._params['dim_embedding']),
            dim_embedding=self._params['dim_embedding'],
            perspective=self._params['perspective'])([x_lt, x_rt])
        x_rt = MultiPerspectiveLayer(
            dim_output=(MultiPerspectiveLayer.num_perspective,
                        self._params['dim_embedding']),
            dim_embedding=self._params['dim_embedding'],
            perspective=self._params['perspective'])([x_rt, x_lt])
        # Aggregation layer.
        x_lt = Bidirectional(
            LSTM(self._params['dim_hidden'],
                 return_sequences=False,
                 return_state=False,
                 kernel_initializer=self._params['w_initializer'],
                 bias_initializer=self._params['b_initializer']),
            merge_mode='concat')(x_lt)
        x_rt = Bidirectional(
            LSTM(self._params['dim_hidden'],
                 return_sequences=False,
                 return_state=False,
                 kernel_initializer=self._params['w_initializer'],
                 bias_initializer=self._params['b_initializer']),
            merge_mode='concat')(x_rt)
        # catenate the forward-backward vector of left & right.
        # Concatenate the concatenated vector of left and right.
        x = Concatenate()([x_lt, x_rt])
        # prediction layer.
        x = Dense(self._params['dim_hidden'],
                  activation=self._params['activation_hidden'])(x)
        x = Dense(self._params['dim_hidden'],
                  activation=self._params['activation_hidden'])(x)
        x_out = self._make_output_layer()(x)
        self._backend = Model(inputs=[input_lt, input_rt],
                              outputs=x_out)
