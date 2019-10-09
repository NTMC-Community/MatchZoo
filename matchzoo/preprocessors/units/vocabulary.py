from .stateful_unit import StatefulUnit


class Vocabulary(StatefulUnit):
    """
    Vocabulary class.

    :param pad_value: The string value for the padding position.
    :param oov_value: The string value for the out-of-vocabulary terms.

    Examples:
        >>> vocab = Vocabulary(pad_value='[PAD]', oov_value='[OOV]')
        >>> vocab.fit(['A', 'B', 'C', 'D', 'E'])
        >>> term_index = vocab.state['term_index']
        >>> term_index  # doctest: +SKIP
        {'[PAD]': 0, '[OOV]': 1, 'D': 2, 'A': 3, 'B': 4, 'C': 5, 'E': 6}
        >>> index_term = vocab.state['index_term']
        >>> index_term  # doctest: +SKIP
        {0: '[PAD]', 1: '[OOV]', 2: 'D', 3: 'A', 4: 'B', 5: 'C', 6: 'E'}

        >>> term_index['out-of-vocabulary-term']
        1
        >>> index_term[0]
        '[PAD]'
        >>> index_term[42]
        Traceback (most recent call last):
            ...
        KeyError: 42
        >>> a_index = term_index['A']
        >>> c_index = term_index['C']
        >>> vocab.transform(['C', 'A', 'C']) == [c_index, a_index, c_index]
        True
        >>> vocab.transform(['C', 'A', '[OOV]']) == [c_index, a_index, 1]
        True
        >>> indices = vocab.transform(list('ABCDDZZZ'))
        >>> ' '.join(vocab.state['index_term'][i] for i in indices)
        'A B C D D [OOV] [OOV] [OOV]'

    """

    def __init__(self, pad_value: str = '<PAD>', oov_value: str = '<OOV>'):
        """Vocabulary unit initializer."""
        super().__init__()
        self._pad = pad_value
        self._oov = oov_value
        self._context['term_index'] = self.TermIndex()
        self._context['index_term'] = dict()

    class TermIndex(dict):
        """Map term to index."""

        def __missing__(self, key):
            """Map out-of-vocabulary terms to index 1."""
            return 1

    def fit(self, tokens: list):
        """Build a :class:`TermIndex` and a :class:`IndexTerm`."""
        self._context['term_index'][self._pad] = 0
        self._context['term_index'][self._oov] = 1
        self._context['index_term'][0] = self._pad
        self._context['index_term'][1] = self._oov
        terms = set(tokens)
        for index, term in enumerate(terms):
            self._context['term_index'][term] = index + 2
            self._context['index_term'][index + 2] = term

    def transform(self, input_: list) -> list:
        """Transform a list of tokens to corresponding indices."""
        return [self._context['term_index'][token] for token in input_]


class BertVocabulary(StatefulUnit):
    """
    Vocabulary class.

    :param pad_value: The string value for the padding position.
    :param oov_value: The string value for the out-of-vocabulary terms.

    Examples:
        >>> vocab = BertVocabulary(pad_value='[PAD]', oov_value='[UNK]')
        >>> indices = vocab.transform(list('ABCDDZZZ'))

    """

    def __init__(self, pad_value: str = '[PAD]', oov_value: str = '[UNK]'):
        """Vocabulary unit initializer."""
        super().__init__()
        self._pad = pad_value
        self._oov = oov_value
        self._context['term_index'] = self.TermIndex()
        self._context['index_term'] = {}

    class TermIndex(dict):
        """Map term to index."""

        def __missing__(self, key):
            """Map out-of-vocabulary terms to index 100 ."""
            return 100

    def fit(self, vocab_path: str):
        """Build a :class:`TermIndex` and a :class:`IndexTerm`."""
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            for idx, line in enumerate(vocab_file):
                term = line.strip()
                self._context['term_index'][term] = idx
                self._context['index_term'][idx] = term

    def transform(self, input_: list) -> list:
        """Transform a list of tokens to corresponding indices."""
        return [self._context['term_index'][token] for token in input_]
