from .stateful_unit import StatefulUnit


class Vocabulary(StatefulUnit):
    """
    Vocabulary class.

    Examples:
        >>> vocab = Vocabulary()
        >>> vocab.fit(['A', 'B', 'C', 'D', 'E'])
        >>> term_index = vocab.state['term_index']
        >>> term_index  # doctest: +SKIP
        {'E': 1, 'C': 2, 'D': 3, 'A': 4, 'B': 5}
        >>> index_term = vocab.state['index_term']
        >>> index_term  # doctest: +SKIP
        {1: 'C', 2: 'A', 3: 'E', 4: 'B', 5: 'D'}

        >>> term_index['out-of-vocabulary-term']
        0
        >>> index_term[0]
        ''
        >>> index_term[42]
        Traceback (most recent call last):
            ...
        KeyError: 42
        >>> a_index = term_index['A']
        >>> c_index = term_index['C']
        >>> vocab.transform(['C', 'A', 'C']) == [c_index, a_index, c_index]
        True
        >>> vocab.transform(['C', 'A', 'OOV']) == [c_index, a_index, 0]
        True
        >>> indices = vocab.transform(list('ABCDDZZZ'))
        >>> ''.join(vocab.state['index_term'][i] for i in indices)
        'ABCDD'

    """

    class IndexTerm(dict):
        """Map index to term."""

        def __missing__(self, key):
            """Map out-of-vocabulary indices to empty string."""
            if key == 0:
                return ''
            else:
                raise KeyError(key)

    class TermIndex(dict):
        """Map term to index."""

        def __missing__(self, key):
            """Map out-of-vocabulary terms to index 0."""
            return 0

    def fit(self, tokens: list):
        """Build a :class:`TermIndex` and a :class:`IndexTerm`."""
        self._state['term_index'] = self.TermIndex()
        self._state['index_term'] = self.IndexTerm()
        terms = set(tokens)
        for index, term in enumerate(terms):
            self._state['term_index'][term] = index + 1
            self._state['index_term'][index + 1] = term

    def transform(self, input_: list) -> list:
        """Transform a list of tokens to corresponding indices."""
        return [self._state['term_index'][token] for token in input_]
