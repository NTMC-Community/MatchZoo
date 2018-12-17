"""An implementation of MultiPerspectiveLayer for Bimpm model."""

from keras import layers
from keras import backend as K
from keras.engine import Layer

from matchzoo import utils
from matchzoo.layers.attention_layer import attention_func


class MultiPerspectiveLayer(Layer):
    """
    A keras implementation of Bimpm multi-perspective layer.

    For detailed information, see Bilateral Multi-Perspective
    Matching for Natural Language Sentences, section 3.2.
    """

    def __init__(
            self,
            dim_input: int,
            dim_perspective: int,
            perspective: dict,
            **kwargs
    ):
        """
        Class initialization.

        :param output_dim: dimensionality of output space.
        """
        self._dim_input = dim_input
        self._dim_perspective = dim_perspective
        self._perspective = perspective
        super(MultiPerspectiveLayer, self).__init__(**kwargs)

    @classmethod
    def list_available_perspectives(cls) -> list:
        """List available strategy for multi-perspective matching."""
        return ['full', 'max-pooling', 'attentive', 'max-attentive']

    @property
    def num_perspective(cls):
        """Get the number of perspectives that is True."""
        return sum(cls._perspective.values())

    def build(self, input_shape: list):
        """Input shape."""
        # The shape of the weights is l * d.
        if self._perspective.get('full'):
            self.full = self.add_weight(name='pool',
                                        shape=(self._dim_perspective,
                                               self._dim_input),
                                        initializer='uniform',
                                        trainable=True)
        if self._perspective.get('max-pooling'):
            self.maxpooling = self.add_weight(name='max-pooling',
                                              shape=(self._dim_perspective,
                                                     self._dim_input),
                                              initializer='uniform',
                                              trainable=True)
        if self._perspective.get('attentive'):
            self.attentive = self.add_weight(name='attentive',
                                             shape=(self._dim_perspective,
                                                    self._dim_input),
                                             initializer='uniform',
                                             trainable=True)
        if self._perspective.get('max-attentive'):
            self.max_attentive = self.add_weight(name='max-attentive',
                                                 shape=(self._dim_perspective,
                                                        self._dim_input),
                                                 initializer='uniform',
                                                 trainable=True)
        self.built = True

    def call(self, x: list, **kwargs):
        """Call."""
        rv = []
        seq_lt, seq_rt = x[:5], x[5:]

        # unpack seq_left and seq_right
        # all hidden states, last hidden state of forward pass,
        # last cell state of forward pass, last hidden state of
        # backward pass, last cell state of backward pass.
        lstm_lt, forward_h_lt, _, backward_h_lt, _ = seq_lt
        lstm_rt, forward_h_rt, _, backward_h_rt, _ = seq_rt

        if self._perspective.get('full'):
            # each forward & backward contextual embedding compare
            # with the last step of the last time step of the other sentence.

            # TODO(tjf): add mask
            h_rt = K.concatenate([forward_h_rt, backward_h_rt], axis=-1)
            full_matching = self._match_tensors_with_tensor(lstm_lt, h_rt, self.full)
            rv.append(full_matching)

        if self._perspective.get('max-pooling'):
            # each contextual embedding compare with each contextual embedding.
            # retain the maximum of each dimension.

            # [batch, time_steps(q), time_steps(p), num_perspective]
            # TODO(tjf): add mask
            match_matrix = self._match_tensors(lstm_lt, lstm_rt, self.maxpooling)
            maxpooling_matching = K.max(match_matrix, axis=2)
            rv.append(maxpooling_matching)

        if self._perspective.get('attentive'):
            # each contextual embedding compare with each contextual embedding.
            # retain sum of weighted mean of each dimension.

            # TODO(tjf): add mask
            att_lt = self.attention(lstm_lt, lstm_rt, pooling='max')
            attentive_matching = self._match_tensors_with_attentive_tensor(lstm_lt, att_lt, self.attentive)
            rv.append(attentive_matching)

        if self._perspective.get('max-attentive'):
            # each contextual embedding compare with each contextual embedding.
            # retain max of weighted mean of each dimension.

            # TODO(tjf): add mask
            att_lt = self.attention(lstm_lt, lstm_rt, pooling='max')
            max_attentive_matching = self._match_tensors_with_attentive_tensor(lstm_lt, att_lt, self.max_attentive)
            rv.append(max_attentive_matching)

        mp_tensor = K.concatenate(rv, axis=-1)

        return mp_tensor

    def attention(self, lstm_lt, lstm_rt, pooling='sum'):
        """
        TODO(tjf) add bilinear attention or mlp attention
        calculate attention
        :param lstm_lt: [batch, steps_lt, d]
        :param lstm_rt: [batch, steps_rt, d]
        :param pooling: sum / max
        :return: [batch, steps_lt, d]
        """
        # [batch, steps_lt, steps_rt]
        atten_score = attention_func([lstm_lt, lstm_rt])
        atten_score = K.expand_dims(atten_score, axis=-1)  # [batch, steps_lt, steps_rt, -1]
        lstm_rt = K.expand_dims(lstm_rt, axis=-1)  # [batch, 1, steps_rt, d]
        att_lt = K.sum(atten_score * lstm_rt, axis=2)
        return att_lt

    def _match_tensors_with_tensor(self, lstm_lt, h_rt, W):
        """
        TODO(tjf): add mask
        """
        # W: -> [1, 1, l, d]
        W = K.expand_dims(W, 0)
        W = K.expand_dims(W, 0)
        # lstm_lt: -> [batch, steps_lt, l, d]
        lstm_lt = W * K.expand_dims(lstm_lt, 2)

        # h_rt: -> [batch, 1, 1, d]
        h_rt = K.expand_dims(h_rt, 1)
        h_rt = K.expand_dims(h_rt, 1)
        h_rt = h_rt

        # matching: -> [batch, steps_lt, l]
        matching = self._cosine_distance(lstm_lt, h_rt, cosine_norm=False)
        return matching

    def _match_tensors_with_attentive_tensor(self, lstm_lt, att_lt, W):
        # TODO(tjf): add mask

        # W: -> [1, 1, l, d]
        W = K.expand_dims(W, 0)
        W = K.expand_dims(W, 0)

        # lstm_lt: -> [batch, steps_lt, l, d]
        lstm_lt = W * K.expand_dims(lstm_lt, 2)

        # att_lt: -> [batch, steps_lt, 1, d]
        att_lt = K.expand_dims(att_lt, 2)

        # matching: -> [batch, steps_lt, l]
        matching = self._cosine_distance(lstm_lt, att_lt, cosine_norm=False)
        return matching

    def _match_tensors(self, lstm_lt, lstm_rt, W):
        """
        TODO(tjf): add mask
        """
        # W: [1, 1, 1, num_perspective, d]
        W = K.expand_dims(W, axis=0)
        W = K.expand_dims(W, axis=0)
        W = K.expand_dims(W, axis=0)

        # lstm_lt: [batch, steps_lt, 1, 1, d]
        lstm_lt = K.expand_dims(lstm_lt, axis=2)
        lstm_lt = K.expand_dims(lstm_lt, axis=2)

        # lstm_rt: [batch, 1, steps_rt, 1, d]
        lstm_rt = K.expand_dims(lstm_rt, axis=2)
        lstm_rt = K.expand_dims(lstm_rt, axis=1)

        # lstm_lt * W: [batch, steps_lt, 1, num_perspective, d]
        # lstm_rt: [batch, 1, steps_rt, 1, 1, d]
        # matching -> [batch, steps_lt, steps_rt, num_perspective]
        matching = self._cosine_distance(lstm_lt * W, lstm_rt, cosine_norm=False)
        return matching

    def _cosine_distance(self, v1, v2, cosine_norm=True, eps=1e-6):
        """
        only requires `K.sum(v1 * v2, axis=-1)`
        """
        # cosine_norm = True
        # v1 [batch, time_steps(v1), 1, l, d]
        # v2 [batch, 1, time_steps(v2), l, d]
        # [batch, time_steps(v1), time_steps(v2), l]
        cosine_numerator = K.sum(v1 * v2, axis=-1)
        if not cosine_norm:
            return K.tanh(cosine_numerator)
        v1_norm = K.sqrt(K.maximum(K.sum(K.square(v1), axis=-1), eps))
        v2_norm = K.sqrt(K.maximum(K.sum(K.square(v2), axis=-1), eps))
        return cosine_numerator / v1_norm / v2_norm

    def compute_output_shape(self, input_shape: list):
        """Compute output shape."""
        shape_a = input_shape[0]
        return (shape_a[0], shape_a[1], self._dim_perspective*len(self._perspective))
