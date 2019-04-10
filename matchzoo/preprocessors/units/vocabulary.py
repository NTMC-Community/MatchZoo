from .stateful_unit import StatefulUnit


class Vocabulary(StatefulUnit):
    """
    Vocabulary class.

    Examples:
        >>> vocab = Vocabulary()
        >>> vocab.fit(['A', 'B', 'C', 'D', 'E'])
        >>> term_index = vocab.state['term_index']
        >>> term_index
        {'<PAD>': 0, '<OOV>': 1, 'D': 2, 'A': 3, 'B': 4, 'C': 5, 'E': 6}
        >>> index_term = vocab.state['index_term']
        >>> index_term
        {0: '<PAD>', 1: '<OOV>', 2: 'D', 3: 'A', 4: 'B', 5: 'C', 6: 'E'}

        >>> term_index['out-of-vocabulary-term']
        1
        >>> index_term[0]
        '<PAD>'
        >>> index_term[42]
        Traceback (most recent call last):
            ...
        KeyError: 42
        >>> a_index = term_index['A']
        >>> c_index = term_index['C']
        >>> vocab.transform(['C', 'A', 'C']) == [c_index, a_index, c_index]
        True
        >>> vocab.transform(['C', 'A', '<OOV>']) == [c_index, a_index, 1]
        True
        >>> indices = vocab.transform(list('ABCDDZZZ'))
        >>> ' '.join(vocab.state['index_term'][i] for i in indices)
        'A B C D D <OOV> <OOV> <OOV>'

    """

    class TermIndex(dict):
        """Map term to index."""

        def __missing__(self, key):
            """Map out-of-vocabulary terms to index 1."""
            return 1

    def fit(self, tokens: list):
        """Build a :class:`TermIndex` and a :class:`IndexTerm`."""
        self._state['term_index'] = self.TermIndex()
        self._state['index_term'] = dict()
        self._state['term_index']['<PAD>'] = 0
        self._state['term_index']['<OOV>'] = 1
        self._state['index_term'][0] = '<PAD>'
        self._state['index_term'][1] = '<OOV>'
        terms = set(tokens)
        for index, term in enumerate(terms):
            self._state['term_index'][term] = index + 2
            self._state['index_term'][index + 2] = term

    def transform(self, input_: list) -> list:
        """Transform a list of tokens to corresponding indices."""
        return [self._state['term_index'][token] for token in input_]
