"""An implementation of DRMMTKS Model."""
from matchzoo import engine
import typing
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Lambda, Activation, \
    Dot, Flatten
from keras.activations import softmax

def show_tensor_info(name, input):
    print('[Layer]: %s\t[Shape]: %s\n' % (name, input.get_shape().as_list()))

class DRMMTKSModel(engine.BaseModel):
    """
    DRMMTKS Model.

    Examples:
        >>> model = DRMMTKSModel()
        >>> model.guess_and_fill_missing_params()
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params()
        params['optimizer'] = 'adam'
        params['input_shapes'] = [(5,), (300,)]
        params.add(engine.Param('trainable_embedding', False))
        params.add(engine.Param('embedding_dim', 300))
        params.add(engine.Param('vocab_size', 100))
        params.add(engine.Param('top_k', 10))
        params.add(engine.Param('hidden_sizes', [5, 1]))
        params.add(engine.Param('embedding_mat', None))
        params.add(engine.Param('embedding_random_scale', 0.2))
        return params

    @property
    def embedding_mat(self) -> np.ndarray:
        """Get pretrained embedding for ArcI model."""
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
        Set pretrained embedding for ArcI model.

        :param embedding_mat: pretrained embedding in numpy format.
        """
        self._params['embedding_mat'] = embedding_mat
        self._params['vocab_size'], self._params['embedding_dim'] = \
            embedding_mat.shape

    def attention_layer(self, input: typing.Any,
                        attention_mask: typing.Any = None):
        """Performs attention on the input."""
        # shape = [B, L, 1]
        dense_input = Dense(1, use_bias=False)(input)
        if attention_mask is not None:
            # Since attention_mask is 1.0 for positions we want to attend and
            # 0.0 for masked positions, this operation will create a tensor
            # which is 0.0 for positions we want to attend and -10000.0 for
            # masked positions.

            # shape = [B, L, 1]
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -1000.0
            # shape = [B, L, 1]
            dense_input += adder
        # shape = [B, L, 1]
        attention_probs = Lambda(lambda x: softmax(x, axis=1),
                                 output_shape = self._params['input_shapes'][0]
                                 )(dense_input)
        return attention_probs

    def multi_layer_perceptron(self, input: typing.Any):
        """Multiple Layer Perceptron."""
        # shape = [B, L, H]
        out = input
        for idx, hidden_size in enumerate(self._params['hidden_sizes']):
            out = Dense(hidden_size)(out)
            out = Activation('tanh')(out)
        # shape = [B, L, 1]
        return out

    def build(self):
        """ Build model structure."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   K = size of top-k

        # Left input and right input.
        # shape = [B, L]
        input_left = Input(name='text_left',
                           shape=self._params['input_shapes'][0])
        show_tensor_info("input_left", input_left)
        # shape = [B, R]
        input_right = Input(name='text_right',
                            shape=self._params['input_shapes'][1])
        show_tensor_info("input_right", input_right)
        # shape = [B, 1]
        """
        length_left = Input(name='length_left',
                             shape=self._params['input_shapes'][2])
        # shape = [B, 1]
        length_right = Input(name='length_right',
                            shape=self._params['input_shapes'][3])
        """
        # Process left input.
        embedding = Embedding(self._params['vocab_size'],
                              self._params['embedding_dim'],
                              weights=[self.embedding_mat],
                              trainable=self._params['trainable_embedding'])
        # shape = [B, L, D]
        embed_left = embedding(input_left)
        show_tensor_info("embed_left", embed_left)
        # shape = [B, R, D]
        embed_right = embedding(input_right)
        show_tensor_info("embed_right", embed_right)
        # shape = [B, L, 1]
        attention_probs = self.attention_layer(embed_left)
        show_tensor_info("attention_probs", attention_probs)

        # Matching histogram of top-k
        # shape = [B, L, R]
        matching_matrix = Dot(axes=[2, 2], normalize=True)([embed_left,
                                                            embed_right])
        show_tensor_info("matching_matrix", matching_matrix)
        # shape = [B, L, K]
        matching_topk = Lambda(lambda x: K.tf.nn.top_k(x,
                                                       k=self._params['top_k'],
                                                       sorted=True)[0]
                               )(matching_matrix)
        show_tensor_info("matching_topk", matching_topk)

        # Process right input.
        # shape = [B, L, 1]
        dense_output = self.multi_layer_perceptron(matching_topk)
        show_tensor_info("dense_output", dense_output)

        # shape = [B, 1, 1]
        dot_score = Dot(axes=[1, 1])([attention_probs, dense_output])
        show_tensor_info("dot_score", dot_score)

        flatten_score = Flatten()(dot_score)
        show_tensor_info("flatten_score", flatten_score)

        x_out = self._make_output_layer()(flatten_score)
        self._backend = Model(
            inputs=[input_left, input_right],
            outputs=x_out)
