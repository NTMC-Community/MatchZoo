import numpy as np

import matchzoo as mz
from matchzoo.data_generator.callbacks import Callback


class Histogram(Callback):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        bin_size: int = 30,
        hist_mode: str = 'CH',
    ):
        self._match_hist_unit = mz.processor_units.MatchingHistogramUnit(
            bin_size=bin_size,
            embedding_matrix=embedding_matrix,
            normalize=True,
            mode=hist_mode
        )

    def on_batch_unpacked(self, x, y):
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
    match_hist_unit: mz.processor_units.MatchingHistogramUnit
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
