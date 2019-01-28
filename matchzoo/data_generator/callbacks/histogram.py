import numpy as np

import matchzoo as mz
from matchzoo.data_generator.callbacks import Callback


class Histogram(Callback):
    """
    Generate data with matching histogram.

    :param embedding_matrix: The embedding matrix used to generator match
                             histogram.
    :param bin_size: The number of bin size of the histogram.
    :param hist_mode: The mode of the :class:`MatchingHistogramUnit`, one of
                     `CH`, `NH`, and `LCH`.
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        bin_size: int = 30,
        hist_mode: str = 'CH',
    ):
        """Init."""
        self._match_hist_unit = mz.preprocessors.units.MatchingHistogram(
            bin_size=bin_size,
            embedding_matrix=embedding_matrix,
            normalize=True,
            mode=hist_mode
        )

    def on_batch_unpacked(self, x, y):
        """Insert `match_histogram` to `x`."""
        x['match_histogram'] = _build_match_histogram(x, self._match_hist_unit)


def _trunc_text(input_text: list, length: list) -> list:
    """
    Truncating the input text according to the input length.

    :param input_text: The input text need to be truncated.
    :param length: The length used to truncated the text.
    :return: The truncated text.
    """
    return [row[:length[idx]] for idx, row in enumerate(input_text)]


def _build_match_histogram(
    x: dict,
    match_hist_unit: mz.preprocessors.units.MatchingHistogram
) -> np.ndarray:
    """
    Generate the matching hisogram for input.

    :param x: The input `dict`.
    :param match_hist_unit: The histogram unit :class:`MatchingHistogramUnit`.
    :return: The matching histogram.
    """
    match_hist = []
    text_left = x['text_left'].tolist()
    text_right = _trunc_text(x['text_right'].tolist(),
                             x['length_right'].tolist())
    for pair in zip(text_left, text_right):
        match_hist.append(match_hist_unit.transform(list(pair)))
    return np.asarray(match_hist)
