"""An implementation of MultiPerspectiveLayer for Bimpm model."""

import tensorflow as tf
from keras import backend as K
from keras.engine import Layer

from matchzoo.contrib.layers.attention_layer import AttentionLayer


class MultiPerspectiveLayer(Layer):
    """
    A keras implementation of multi-perspective layer of BiMPM.

    For detailed information, see Bilateral Multi-Perspective
    Matching for Natural Language Sentences, section 3.2.

    Examples:
        >>> import matchzoo as mz
        >>> perspective={'full': True, 'max-pooling': True,
        ...     'attentive': True, 'max-attentive': True}
        >>> layer = mz.contrib.layers.MultiPerspectiveLayer(
        ...     att_dim=50, mp_dim=20, perspective=perspective)
        >>> layer.compute_output_shape(
        ...     [(32, 10, 100), (32, 50), None, (32, 50), None,
        ...     [(32, 40, 100), (32, 50), None, (32, 50), None]])
        (32, 10, 83)

    """

    def __init__(self,
                 att_dim: int,
                 mp_dim: int,
                 perspective: dict):
        """Class initialization."""
        super(MultiPerspectiveLayer, self).__init__()
        self._att_dim = att_dim
        self._mp_dim = mp_dim
        self._perspective = perspective

    @classmethod
    def list_available_perspectives(cls) -> list:
        """List available strategy for multi-perspective matching."""
        return ['full', 'max-pooling', 'attentive', 'max-attentive']

    @property
    def num_perspective(self):
        """Get the number of perspectives that is True."""
        return sum(self._perspective.values())

    def build(self, input_shape: list):
        """Input shape."""
        # The shape of the weights is l * d.
        if self._perspective.get('full'):
            self.full_match = MpFullMatch(self._mp_dim)

        if self._perspective.get('max-pooling'):
            self.max_pooling_match = MpMaxPoolingMatch(self._mp_dim)

        if self._perspective.get('attentive'):
            self.attentive_match = MpAttentiveMatch(self._att_dim,
                                                    self._mp_dim)

        if self._perspective.get('max-attentive'):
            self.max_attentive_match = MpMaxAttentiveMatch(self._att_dim)
        self.built = True

    def call(self, x: list, **kwargs):
        """Call."""
        seq_lt, seq_rt = x[:5], x[5:]
        # unpack seq_left and seq_right
        # all hidden states, last hidden state of forward pass,
        # last cell state of forward pass, last hidden state of
        # backward pass, last cell state of backward pass.
        lstm_reps_lt, forward_h_lt, _, backward_h_lt, _ = seq_lt
        lstm_reps_rt, forward_h_rt, _, backward_h_rt, _ = seq_rt

        match_tensor_list = []
        match_dim = 0
        if self._perspective.get('full'):
            # Each forward & backward contextual embedding compare
            # with the last step of the last time step of the other sentence.
            h_lt = tf.concat([forward_h_lt, backward_h_lt], axis=-1)
            full_match_tensor = self.full_match([h_lt, lstm_reps_rt])
            match_tensor_list.append(full_match_tensor)
            match_dim += self._mp_dim + 1

        if self._perspective.get('max-pooling'):
            # Each contextual embedding compare with each contextual embedding.
            # retain the maximum of each dimension.
            max_match_tensor = self.max_pooling_match([lstm_reps_lt,
                                                       lstm_reps_rt])
            match_tensor_list.append(max_match_tensor)
            match_dim += self._mp_dim

        if self._perspective.get('attentive'):
            # Each contextual embedding compare with each contextual embedding.
            # retain sum of weighted mean of each dimension.
            attentive_tensor = self.attentive_match([lstm_reps_lt,
                                                     lstm_reps_rt])
            match_tensor_list.append(attentive_tensor)
            match_dim += self._mp_dim + 1

        if self._perspective.get('max-attentive'):
            # Each contextual embedding compare with each contextual embedding.
            # retain max of weighted mean of each dimension.
            relevancy_matrix = _calc_relevancy_matrix(lstm_reps_lt,
                                                      lstm_reps_rt)
            max_attentive_tensor = self.max_attentive_match([lstm_reps_lt,
                                                             lstm_reps_rt,
                                                             relevancy_matrix])
            match_tensor_list.append(max_attentive_tensor)
            match_dim += self._mp_dim + 1

        mp_tensor = tf.concat(match_tensor_list, axis=-1)
        return mp_tensor

    def compute_output_shape(self, input_shape: list):
        """Compute output shape."""
        shape_a = input_shape[0]

        match_dim = 0
        if self._perspective.get('full'):
            match_dim += self._mp_dim + 1
        if self._perspective.get('max-pooling'):
            match_dim += self._mp_dim
        if self._perspective.get('attentive'):
            match_dim += self._mp_dim + 1
        if self._perspective.get('max-attentive'):
            match_dim += self._mp_dim + 1

        return shape_a[0], shape_a[1], match_dim


class MpFullMatch(Layer):
    """Mp Full Match Layer."""

    def __init__(self, mp_dim):
        """Init."""
        super(MpFullMatch, self).__init__()
        self.mp_dim = mp_dim

    def build(self, input_shapes):
        """Build."""
        # input_shape = input_shapes[0]
        self.built = True

    def call(self, x, **kwargs):
        """Call.
        """
        rep_lt, reps_rt = x
        att_lt = tf.expand_dims(rep_lt, 1)

        match_tensor, match_dim = _multi_perspective_match(self.mp_dim,
                                                           reps_rt,
                                                           att_lt)
        # match_tensor => [b, len_rt, mp_dim+1]
        return match_tensor

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape[1][0], input_shape[1][1], self.mp_dim + 1


class MpMaxPoolingMatch(Layer):
    """MpMaxPoolingMatch."""

    def __init__(self, mp_dim):
        """Init."""
        super(MpMaxPoolingMatch, self).__init__()
        self.mp_dim = mp_dim

    def build(self, input_shapes):
        """Build."""
        d = input_shapes[0][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, 1, 1, self.mp_dim, d),
                                      initializer='uniform',
                                      trainable=True)
        self.built = True

    def call(self, x, **kwargs):
        """Call."""
        reps_lt, reps_rt = x

        # kernel: [1, 1, 1, mp_dim, d]
        # lstm_lt => [b, len_lt, 1, 1, d]
        reps_lt = tf.expand_dims(reps_lt, axis=2)
        reps_lt = tf.expand_dims(reps_lt, axis=2)
        reps_lt = reps_lt * self.kernel

        # lstm_rt -> [b, 1, len_rt, 1, d]
        reps_rt = tf.expand_dims(reps_rt, axis=2)
        reps_rt = tf.expand_dims(reps_rt, axis=1)

        match_tensor = _cosine_distance(reps_lt, reps_rt, cosine_norm=False)
        max_match_tensor = tf.reduce_max(match_tensor, axis=1)
        # match_tensor => [b, len_rt, m]
        return max_match_tensor

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape[1][0], input_shape[1][1], self.mp_dim


class MpAttentiveMatch(Layer):
    """
    MpAttentiveMatch Layer.

    Reference:
    https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py#L188-L193

    Examples:
        >>> import matchzoo as mz
        >>> layer = mz.contrib.layers.multi_perspective_layer.MpAttentiveMatch(
        ...     att_dim=30, mp_dim=20)
        >>> layer.compute_output_shape([(32, 10, 100), (32, 40, 100)])
        (32, 40, 20)

    """

    def __init__(self, att_dim, mp_dim):
        """Init."""
        super(MpAttentiveMatch, self).__init__()
        self.att_dim = att_dim
        self.mp_dim = mp_dim

    def build(self, input_shapes):
        """Build."""
        # input_shape = input_shapes[0]
        self.built = True

    def call(self, x, **kwargs):
        """Call."""
        reps_lt, reps_rt = x[0], x[1]
        # attention prob matrix
        attention_layer = AttentionLayer(self.att_dim)
        attn_prob = attention_layer([reps_rt, reps_lt])
        # attention reps
        att_lt = K.batch_dot(attn_prob, reps_lt)
        # mp match
        attn_match_tensor, match_dim = _multi_perspective_match(self.mp_dim,
                                                                reps_rt,
                                                                att_lt)
        return attn_match_tensor

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape[1][0], input_shape[1][1], self.mp_dim


class MpMaxAttentiveMatch(Layer):
    """MpMaxAttentiveMatch."""

    def __init__(self, mp_dim):
        """Init."""
        super(MpMaxAttentiveMatch, self).__init__()
        self.mp_dim = mp_dim

    def build(self, input_shapes):
        """Build."""
        # input_shape = input_shapes[0]
        self.built = True

    def call(self, x):
        """Call."""
        reps_lt, reps_rt = x[0], x[1]
        relevancy_matrix = x[2]
        max_att_lt = cal_max_question_representation(reps_lt, relevancy_matrix)
        max_attentive_tensor, match_dim = _multi_perspective_match(self.mp_dim,
                                                                   reps_rt,
                                                                   max_att_lt)
        return max_attentive_tensor


def cal_max_question_representation(reps_lt, attn_scores):
    """
    Calculate max_question_representation.

    :param reps_lt: [batch_size, passage_len, hidden_size]
    :param attn_scores: []
    :return: [batch_size, passage_len, hidden_size].
    """
    attn_positions = tf.argmax(attn_scores, axis=2)
    max_reps_lt = collect_representation(reps_lt, attn_positions)
    return max_reps_lt


def collect_representation(representation, positions):
    """
    Collect_representation.

    :param representation: [batch_size, node_num, feature_dim]
    :param positions: [batch_size, neighbour_num]
    :return: [batch_size, neighbour_num]?
    """
    return collect_probs(representation, positions)


def collect_final_step_of_lstm(lstm_representation, lengths):
    """
    Collect final step of lstm.

    :param lstm_representation: [batch_size, len_rt, dim]
    :param lengths: [batch_size]
    :return: [batch_size, dim]
    """
    lengths = tf.maximum(lengths, K.zeros_like(lengths))

    batch_size = tf.shape(lengths)[0]
    # shape (batch_size)
    batch_nums = tf.range(0, limit=batch_size)
    # shape (batch_size, 2)
    indices = tf.stack((batch_nums, lengths), axis=1)
    result = tf.gather_nd(lstm_representation, indices,
                            name='last-forwar-lstm')
    # [batch_size, dim]
    return result


def collect_probs(probs, positions):
    """
    Collect Probabilities.

    Reference:
    https://github.com/zhiguowang/BiMPM/blob/master/src/layer_utils.py#L128-L140
    :param probs: [batch_size, chunks_size]
    :param positions: [batch_size, pair_size]
    :return: [batch_size, pair_size]
    """
    batch_size = tf.shape(probs)[0]
    pair_size = tf.shape(positions)[1]
    # shape (batch_size)
    batch_nums = K.arange(0, batch_size)
    # [batch_size, 1]
    batch_nums = tf.reshape(batch_nums, shape=[-1, 1])
    # [batch_size, pair_size]
    batch_nums = K.tile(batch_nums, [1, pair_size])

    # shape (batch_size, pair_size, 2)
    # Alert: to solve error message
    positions = tf.cast(positions, tf.int32)
    indices = tf.stack([batch_nums, positions], axis=2)

    pair_probs = tf.gather_nd(probs, indices)
    # pair_probs = tf.reshape(pair_probs, shape=[batch_size, pair_size])
    return pair_probs


def _multi_perspective_match(mp_dim, reps_rt, att_lt,
                             with_cosine=True, with_mp_cosine=True):
    """
    The core function of zhiguowang's implementation.

    reference:
    https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py#L207-L223
    :param mp_dim: about 20
    :param reps_rt: [batch, len_rt, dim]
    :param att_lt: [batch, len_rt, dim]
    :param with_cosine: True
    :param with_mp_cosine: True
    :return: [batch, len, 1 + mp_dim]
    """
    shape_rt = tf.shape(reps_rt)
    batch_size = shape_rt[0]
    len_lt = shape_rt[1]

    match_dim = 0
    match_result_list = []
    if with_cosine:
        cosine_tensor = _cosine_distance(reps_rt, att_lt, False)
        cosine_tensor = tf.reshape(cosine_tensor,
                                  [batch_size, len_lt, 1])
        match_result_list.append(cosine_tensor)
        match_dim += 1

    if with_mp_cosine:
        mp_cosine_layer = MpCosineLayer(mp_dim)
        mp_cosine_tensor = mp_cosine_layer([reps_rt, att_lt])
        mp_cosine_tensor = tf.reshape(mp_cosine_tensor,
                                     [batch_size, len_lt, mp_dim])
        match_result_list.append(mp_cosine_tensor)
        match_dim += mp_cosine_layer.mp_dim

    match_result = tf.concat(match_result_list, 2)
    return match_result, match_dim


class MpCosineLayer(Layer):
    """
    Implementation of Multi-Perspective Cosine Distance.

    Reference:
    https://github.com/zhiguowang/BiMPM/blob/master/src/match_utils.py#L121-L129

    Examples:
        >>> import matchzoo as mz
        >>> layer = mz.contrib.layers.multi_perspective_layer.MpCosineLayer(
        ...     mp_dim=50)
        >>> layer.compute_output_shape([(32, 10, 100), (32, 10, 100)])
        (32, 10, 50)

    """

    def __init__(self, mp_dim, **kwargs):
        """Init."""
        self.mp_dim = mp_dim
        super(MpCosineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build."""
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, 1, self.mp_dim,
                                             input_shape[0][-1]),
                                      initializer='uniform',
                                      trainable=True)
        super(MpCosineLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        """Call."""
        v1, v2 = x
        v1 = tf.expand_dims(v1, 2) * self.kernel  # [b, s_lt, m, d]
        v2 = tf.expand_dims(v2, 2)  # [b, s_lt, 1, d]
        return _cosine_distance(v1, v2, False)

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape[0][0], input_shape[0][1], self.mp_dim


def _calc_relevancy_matrix(reps_lt, reps_rt):
    reps_lt = tf.expand_dims(reps_lt, 1)  # [b, 1, len_lt, d]
    reps_rt = tf.expand_dims(reps_rt, 2)  # [b, len_rt, 1, d]
    relevancy_matrix = _cosine_distance(reps_lt, reps_rt)
    # => [b, len_rt, len_lt, d]
    return relevancy_matrix


def _mask_relevancy_matrix(relevancy_matrix, mask_lt, mask_rt):
    """
    Mask relevancy matrix.

    :param relevancy_matrix: [b, len_rt, len_lt]
    :param mask_lt: [b, len_lt]
    :param mask_rt: [b, len_rt]
    :return: masked_matrix: [b, len_rt, len_lt]
    """
    if mask_lt is not None:
        relevancy_matrix = relevancy_matrix * tf.expand_dims(mask_lt, 1)
    relevancy_matrix = relevancy_matrix * tf.expand_dims(mask_rt, 2)
    return relevancy_matrix


def _cosine_distance(v1, v2, cosine_norm=True, eps=1e-6):
    """
    Only requires `tf.reduce_sum(v1 * v2, axis=-1)`.

    :param v1: [batch, time_steps(v1), 1, m, d]
    :param v2: [batch, 1, time_steps(v2), m, d]
    :param cosine_norm: True
    :param eps: 1e-6
    :return: [batch, time_steps(v1), time_steps(v2), m]
    """
    cosine_numerator = tf.reduce_sum(v1 * v2, axis=-1)
    if not cosine_norm:
        return K.tanh(cosine_numerator)
    v1_norm = K.sqrt(tf.maximum(tf.reduce_sum(tf.square(v1), axis=-1), eps))
    v2_norm = K.sqrt(tf.maximum(tf.reduce_sum(tf.square(v2), axis=-1), eps))
    return cosine_numerator / v1_norm / v2_norm
