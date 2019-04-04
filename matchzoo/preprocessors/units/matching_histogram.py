import numpy as np

from .unit import Unit


class MatchingHistogram(Unit):
    """
    MatchingHistogramUnit Class.

    :param bin_size: The number of bins of the matching histogram.
    :param embedding_matrix: The word embedding matrix applied to calculate
                             the matching histogram.
    :param normalize: Boolean, normalize the embedding or not.
    :param mode: The type of the historgram, it should be one of 'CH', 'NG',
                 or 'LCH'.

    Examples:
        >>> embedding_matrix = np.array([[1.0, -1.0], [1.0, 2.0], [1.0, 3.0]])
        >>> text_left = [0, 1]
        >>> text_right = [1, 2]
        >>> histogram = MatchingHistogram(3, embedding_matrix, True, 'CH')
        >>> histogram.transform([text_left, text_right])
        [[3.0, 1.0, 1.0], [1.0, 2.0, 2.0]]

    """

    def __init__(self, bin_size: int = 30, embedding_matrix=None,
                 normalize=True, mode: str = 'LCH'):
        """The constructor."""
        self._hist_bin_size = bin_size
        self._embedding_matrix = embedding_matrix
        if normalize:
            self._normalize_embedding()
        self._mode = mode

    def _normalize_embedding(self):
        """Normalize the embedding matrix."""
        l2_norm = np.sqrt(
            (self._embedding_matrix * self._embedding_matrix).sum(axis=1)
        )
        self._embedding_matrix = \
            self._embedding_matrix / l2_norm[:, np.newaxis]

    def transform(self, input_: list) -> list:
        """Transform the input text."""
        text_left, text_right = input_
        matching_hist = np.ones((len(text_left), self._hist_bin_size),
                                dtype=np.float32)
        embed_left = self._embedding_matrix[text_left]
        embed_right = self._embedding_matrix[text_right]
        matching_matrix = embed_left.dot(np.transpose(embed_right))
        for (i, j), value in np.ndenumerate(matching_matrix):
            bin_index = int((value + 1.) / 2. * (self._hist_bin_size - 1.))
            matching_hist[i][bin_index] += 1.0
        if self._mode == 'NH':
            matching_sum = matching_hist.sum(axis=1)
            matching_hist = matching_hist / matching_sum[:, np.newaxis]
        elif self._mode == 'LCH':
            matching_hist = np.log(matching_hist)
        return matching_hist.tolist()
