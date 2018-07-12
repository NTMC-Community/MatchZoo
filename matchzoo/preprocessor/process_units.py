"""Matchzoo toolkit for text pre-processing."""

import re
import abc
import nltk
import typing
import many_stop_words


match_punc = re.compile('[^\w\s]')


class ProcessorUnit(metaclass=abc.ABCMeta):
    """Process unit do not persive state (i.e. do not need fit)."""

    @abc.abstractmethod
    def transform(self, input: typing.Any):
        """Abstract base method, need to be implemented in subclass."""


class StatefulProcessorUnit(ProcessorUnit, metaclass=abc.ABCMeta):
    """Process unit do persive state (i.e. need fit)."""

    def __init__(self):
        """Initialization."""
        self._state = {}

    @property
    def state(self):
        """Get current state."""
        return self._state

    @abc.abstractmethod
    def fit(self, input: typing.Any):
        """Abstract base method, need to be implemented in subclass."""


class TokenizeUnit(ProcessorUnit):
    """Process unit for text tokenization."""

    def transform(self, input: str) -> list:
        """
        Process input data from raw terms to list of tokens.

        :param input: raw textual input.

        :return tokens: tokenized tokens as a list.
        """
        return nltk.word_tokenize(input)


class LowercaseUnit(ProcessorUnit):
    """Process unit for text lower case."""

    def transform(self, tokens: list) -> list:
        """
        Convert list of tokens to lower case.

        :param tokens: list of tokens.

        :return tokens: lower-cased list of tokens.
        """
        return [token.lower() for token in tokens]


class PuncRemovalUnit(ProcessorUnit):
    """Process unit for remove punctuations."""

    def transform(self, tokens: list) -> list:
        """
        Remove punctuations from list of tokens.

        :param tokens: list of toekns.

        :return rv: tokens  without punctuation.
        """
        return [token for token in tokens if not match_punc.search(token)]


class DigitRemovalUnit(ProcessorUnit):
    """Process unit to remove digits."""

    def transform(self, tokens: list) -> list:
        """
        Remove digits from list of tokens.

        :param tokens: list of tokens to be filtered.

        :return tokens: tokens of tokens without digits.
        """
        return [token for token in tokens if not token.isdigit()]


class StopRemovalUnit(ProcessorUnit):
    """Process unit to remove stop words."""

    def __init__(self, lang='en'):
        """Initialization."""
        self._lang = lang

    def transform(self, tokens: list) -> list:
        """
        Remove stopwords from list of tokenized tokens.

        :param tokens: list of tokenized tokens.
        :param lang: language code for stopwords.

        :return tokens: list of tokenized tokens without stopwords.
        """
        return [token
                for token
                in tokens
                if token not in self.get_stopwords()]

    def get_stopwords(self) -> list:
        """
        Get stopwords based on language.

        :params lang: language code.

        :return stop_list: list of stop words.
        """
        return many_stop_words.get_stop_words(self._lang)


class StemmingUnit(ProcessorUnit):
    """Process unit for token stemming."""

    def __init__(self, stemmer='porter'):
        """Initialization."""
        self.stemmer = stemmer

    def transform(self, tokens: list) -> list:
        """
        Reducing inflected words to their word stem, base or root form.

        :param tokens: list of string to be stemmed.
        :param stemmer: stemmer to use, `porter` or `lancaster`.

        :raise ValueError: stemmer type should be porter or lancaster.

        :return tokens: stemmed token.
        """
        if self.stemmer == 'porter':
            porter_stemmer = nltk.stem.PorterStemmer()
            return [porter_stemmer.stem(token) for token in tokens]
        elif self.stemmer == 'lancaster' or self.stemmer == 'krovetz':
            lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
            return [lancaster_stemmer.stem(token) for token in tokens]
        else:
            raise ValueError(
                'Not supported supported stemmer type: {}'.format(
                    self.stemmer))


class LemmatizationUnit(ProcessorUnit):
    """Process unit for token lemmatization."""

    def transform(self, tokens: list) -> list:
        """
        Lemmatization a sequence of tokens.

        :param tokens: list of tokens to be lemmatized.

        :return tokens: list of lemmatizd tokens.
        """
        lemmatizer = nltk.WordNetLemmatizer()
        return [lemmatizer.lemmatize(token, pos='v') for token in tokens]


class NgramLetterUnit(StatefulProcessorUnit):
    """
    Process unit for n-letter generation.

    Triletter is used in :class:`DSSMModel`.
    This processor is expected to execute after `Vocab`
    has been created.

    Returned `input_dim` is the dimensionality of :class:`DSSMModel`.
    """

    def __init__(self):
        """Initialization."""
        super().__init__()

    def _create_n_letters(self, tokens: list, ngram: int=3) -> list:
        """
        Create n_letters.

        For example, `word` should be represented as `#wo`,
        `wor`, `ord` and `rd#`.

        :param tokens: list of tokens to be transformed.
        :param ngram: By default use 3-gram (tri-letter).

        :return n_letters: generated n_letters.
        :return: length of n_letters, dimensionality of :DSSMModel:.
        """
        n_letters = set()
        for token in tokens:
            token = '#' + token + '#'
            while len(token) >= ngram:
                n_letters.add(token[:ngram])
                token = token[1:]
        return n_letters, len(n_letters)

    def transform(self, tokens: list, ngram: int=3) -> list:
        """
        Transform token into tri-letter.

        For example, `word` should be represented as `#wo`,
        `wor`, `ord` and `rd#`.

        :param tokens: list of tokens to be transformed.
        :param ngram: By default use 3-gram (tri-letter).

        :return: set of tri-letters, dependent on `ngram`.
        """
        n_letters, _ = self._create_n_letters(tokens, ngram)
        return n_letters

    def fit(self, tokens: list, ngram: int=3):
        """
        Fiitting parameters (shape of word hashing layer) for :DSSM:.

        :param tokens: list of tokens to be fitted.
        :param ngram: By default use 3-gram (tri-letter).
        """
        _, input_dim = self._create_n_letters(tokens, ngram)
        self._state = {'input_dim': input_dim}


class VocabularyUnit(StatefulProcessorUnit):
    """
    Vocabulary class.

    Examples:
        >>> vocab = VocabularyUnit()
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

        >>> indices = vocab.transform('ABCDDZZZ')
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

    def transform(self, tokens: list) -> list:
        """Transform a list of tokens to corresponding indices."""
        return [self._state['term_index'][token] for token in tokens]
