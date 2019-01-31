from .unit import Unit


class NgramLetter(Unit):
    """
    Process unit for n-letter generation.

    Triletter is used in :class:`DSSMModel`.
    This processor is expected to execute before `Vocab`
    has been created.

    Examples:
        >>> triletter = NgramLetter()
        >>> rv = triletter.transform(['hello', 'word'])
        >>> len(rv)
        9
        >>> rv
        ['#he', 'hel', 'ell', 'llo', 'lo#', '#wo', 'wor', 'ord', 'rd#']
        >>> triletter = NgramLetter(reduce_dim=False)
        >>> rv = triletter.transform(['hello', 'word'])
        >>> len(rv)
        2
        >>> rv
        [['#he', 'hel', 'ell', 'llo', 'lo#'], ['#wo', 'wor', 'ord', 'rd#']]

    """

    def __init__(self, ngram: int = 3, reduce_dim: bool = True):
        """
        Class initialization.

        :param ngram: By default use 3-gram (tri-letter).
        :param reduce_dim: Reduce to 1-D list for sentence representation.
        """
        self._ngram = ngram
        self._reduce_dim = reduce_dim

    def transform(self, input_: list) -> list:
        """
        Transform token into tri-letter.

        For example, `word` should be represented as `#wo`,
        `wor`, `ord` and `rd#`.

        :param input_: list of tokens to be transformed.

        :return n_letters: generated n_letters.
        """
        n_letters = []
        for token in input_:
            token = '#' + token + '#'
            token_ngram = []
            while len(token) >= self._ngram:
                token_ngram.append(token[:self._ngram])
                token = token[1:]
            if self._reduce_dim:
                n_letters.extend(token_ngram)
            else:
                n_letters.append(token_ngram)
        return n_letters
