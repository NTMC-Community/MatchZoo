import collections
import typing

import numpy as np

from .stateful_unit import StatefulUnit


class FrequencyFilter(StatefulUnit):
    """
    Frequency filter unit.

    :param low: Lower bound, inclusive.
    :param high: Upper bound, exclusive.
    :param mode: One of `tf` (term frequency), `df` (document frequency),
        and `idf` (inverse document frequency).

    Examples::
        >>> import matchzoo as mz

    To filter based on term frequency (tf):
        >>> tf_filter = mz.preprocessors.units.FrequencyFilter(
        ...     low=2, mode='tf')
        >>> tf_filter.fit([['A', 'B', 'B'], ['C', 'C', 'C']])
        >>> tf_filter.transform(['A', 'B', 'C'])
        ['B', 'C']

    To filter based on document frequency (df):
        >>> tf_filter = mz.preprocessors.units.FrequencyFilter(
        ...     low=2, mode='df')
        >>> tf_filter.fit([['A', 'B'], ['B', 'C']])
        >>> tf_filter.transform(['A', 'B', 'C'])
        ['B']

    To filter based on inverse document frequency (idf):
        >>> idf_filter = mz.preprocessors.units.FrequencyFilter(
        ...     low=1.2, mode='idf')
        >>> idf_filter.fit([['A', 'B'], ['B', 'C', 'D']])
        >>> idf_filter.transform(['A', 'B', 'C'])
        ['A', 'C']

    """

    def __init__(self, low: float = 0, high: float = float('inf'),
                 mode: str = 'df'):
        """Frequency filter unit."""
        super().__init__()
        self._low = low
        self._high = high
        self._mode = mode

    def fit(self, list_of_tokens: typing.List[typing.List[str]]):
        """Fit `list_of_tokens` by calculating `mode` states."""
        valid_terms = set()
        if self._mode == 'tf':
            stats = self._tf(list_of_tokens)
        elif self._mode == 'df':
            stats = self._df(list_of_tokens)
        elif self._mode == 'idf':
            stats = self._idf(list_of_tokens)
        else:
            raise ValueError(f"{self._mode} is not a valid filtering mode."
                             f"Mode must be one of `tf`, `df`, and `idf`.")

        for k, v in stats.items():
            if self._low <= v < self._high:
                valid_terms.add(k)

        self._state[self._mode] = valid_terms

    def transform(self, input_: list) -> list:
        """Transform a list of tokens by filtering out unwanted words."""
        valid_terms = self._state[self._mode]
        return list(filter(lambda token: token in valid_terms, input_))

    @classmethod
    def _tf(cls, list_of_tokens: list) -> dict:
        stats = collections.Counter()
        for tokens in list_of_tokens:
            stats.update(tokens)
        return stats

    @classmethod
    def _df(cls, list_of_tokens: list) -> dict:
        stats = collections.Counter()
        for tokens in list_of_tokens:
            stats.update(set(tokens))
        return stats

    @classmethod
    def _idf(cls, list_of_tokens: list) -> dict:
        num_docs = len(list_of_tokens)
        stats = cls._df(list_of_tokens)
        for key, val in stats.most_common():
            stats[key] = np.log((1 + num_docs) / (1 + val)) + 1
        return stats
