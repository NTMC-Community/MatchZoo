"""Matchzoo toolkit for text pre-processing."""

import re
import abc
import nltk
import typing
import collections
import numpy as np

match_punc = re.compile(r'[^\w\s]')


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
    """
    Process unit to remove stop words.

    Example:
        >>> unit = StopRemovalUnit()
        >>> unit.transform(['a', 'the', 'test'])
        ['test']
        >>> type(unit.stopwords)
        <class 'list'>
    """

    def __init__(self, lang: str='english'):
        """Initialization."""
        self._lang = lang
        self._stop = nltk.corpus.stopwords.words(self._lang)

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
                if token not in self._stop]

    @property
    def stopwords(self) -> list:
        """
        Get stopwords based on language.

        :params lang: language code.
        :return: list of stop words.
        """
        return self._stop


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


class NgramLetterUnit(ProcessorUnit):
    """
    Process unit for n-letter generation.

    Triletter is used in :class:`DSSMModel`.
    This processor is expected to execute before `Vocab`
    has been created.

    Examples:
        >>> triletter = NgramLetterUnit()
        >>> rv = triletter.transform(['word'])
        >>> len(rv)
        4
        >>> rv
        ['#wo', 'wor', 'ord', 'rd#']

    """

    def transform(self, tokens: list, ngram: int=3) -> list:
        """
        Transform token into tri-letter.

        For example, `word` should be represented as `#wo`,
        `wor`, `ord` and `rd#`.

        :param tokens: list of tokens to be transformed.
        :param ngram: By default use 3-gram (tri-letter).

        :return n_letters: generated n_letters.
        """
        n_letters = []
        for token in tokens:
            token = '#' + token + '#'
            while len(token) >= ngram:
                n_letters.append(token[:ngram])
                token = token[1:]
        return n_letters


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
        # convert high dimension tokens to 1d tokens
        tokens = np.array(tokens).flatten().tolist()
        terms = set(tokens)
        for index, term in enumerate(terms):
            self._state['term_index'][term] = index + 1
            self._state['index_term'][index + 1] = term

    def transform(self, tokens: list) -> list:
        """Transform a list of tokens to corresponding indices."""
        return [self._state['term_index'][token] for token in tokens]


class WordHashingUnit(ProcessorUnit):
    """
    Create word-hashing layer for :class:`DSSMModel`.

    The input of :class:`WordHashingUnit` should be a list
    of `tri-letters` extracted from one document. The output
    of :class:`WordHashingUnit` is the word-hashing representation
    of this document.

    :class:`NgramLetterUnit` and :class:`VocabularyUnit` are two
    essential prerequisite of :class:`WordHashingUnit`.

    TODO Update :class:`WordHashingUnit` to generalize more `DSSM`
    like models such as `CDSSM` and `LSTM-DSSM`.

    Examples:
       >>> tri_letters = ['#te', 'tes','est', 'st#', 'oov']
       >>> word_hashing = WordHashingUnit(
       ...     term_index={'': 0,'st#': 1, '#te': 2, 'est': 3, 'tes': 4})
       >>> hashing = word_hashing.transform(tri_letters)
       >>> hashing[0]
       1.0
       >>> hashing[1]
       1.0
       >>> hashing.shape
       (6,)

    """

    def __init__(
        self,
        term_index: dict,
    ):
        """
        Class initialization.

        :param term_index: term-index mapping generated by
            :class:`VocabularyUnit`.
        :param dim_triletter: dimensionality of tri_leltters.
        """
        self._term_index = term_index

    def transform(
        self,
        tri_letters: list
    ) -> np.ndarray:
        """
        Transform list of :attr:`tri-letters` into word hashing layer.

        :param tri_letters: list of `tri_letters` generated by
            :class:`NgramLetterUnit`.
        :return: Word hashing representation of `tri-letters`.
        """
        hashing = np.zeros(len(self._term_index) + 1)
        counted_tri_letters = collections.Counter(tri_letters)
        for key, value in counted_tri_letters.items():
            idx = self._term_index.get(key, 0)
            hashing[idx] = value
        return hashing


class SlidingWindowUnit(ProcessorUnit):
    """
    SlidingWindowUnit class.

    Used to get information from a window sliding
    through the input data.

    Examples:
        >>> data = [[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
        >>> sliding = SlidingWindowUnit()
        >>> output = sliding.transform(data)
        >>> output[0]
        array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    """

    def __init__(self, sliding_window: int=3):
        """
        Class initialization.

        :param sliding_window: sliding window length.
        """
        self._sliding_window = sliding_window

    def transform(self, inputs: list) -> list:
        """
        Compute the result of a window sliding through input data.

        For short inputs that can not fill :attr:`sliding_window`, the
        unit return an empty array.

        :param inputs: sequential input data.
        :return: window sliding result.
        """
        output = []
        inputs = np.array(inputs)
        if len(inputs.shape) == 1:
            inputs = np.expand_dims(inputs, -1)
        while len(inputs) >= self._sliding_window:
            output.append(np.concatenate(
                inputs[:self._sliding_window], axis=-1))
            inputs = inputs[1:]
        return np.array(output).tolist()


class FixedLengthUnit(ProcessorUnit):
    """
    FixedLengthUnit Class.

    Process unit to get the fixed length text.

    Examples:
        >>> fixedlen = FixedLengthUnit(3)
        >>> fixedlen.transform(range(1, 6)) == [3, 4, 5]
        True
        >>> fixedlen = FixedLengthUnit(3)
        >>> fixedlen.transform(range(1, 3)) == [0, 1, 2]
        True

    """

    def __init__(self, text_length: int, pad_value: int=0,
                 pad_mode: str='pre', truncate_mode: str='pre'):
        """
        Class initialization.

        :param text_length: fixed length of the text.
        :param pad_value: if text length is smaller than :attr:`text_length`,
            filling text with :attr:`pad_value`.
        :param pad_mode: String, `pre` or `post`:
            pad either before or after each sequence.
        :param truncate_mode: String, `pre` or `post`:
            remove values from sequences larger than :attr:`text_length`,
            either at the beginning or at the end of the sequences.
        """
        self._text_length = text_length
        self._pad_value = pad_value
        self._pad_mode = pad_mode
        self._truncate_mode = truncate_mode

    def transform(self, tokens: list) -> list:
        """
        Transform list of tokenized tokens into the fixed length text.

        :param tokens: list of tokenized tokens.

        :return tokens: list of tokenized tokens in fixed length.
        """
        np_tokens = np.array(tokens)
        fixed_tokens = np.full([self._text_length], self._pad_value,
                               dtype=np_tokens.dtype)

        if self._truncate_mode == 'pre':
            trunc_tokens = tokens[-self._text_length:]
        elif self._truncate_mode == 'post':
            trunc_tokens = tokens[:self._text_length]
        else:
            raise ValueError('{} is not a vaild '
                             'truncate mode.'.format(self._truncate_mode))

        if self._pad_mode == 'post':
            fixed_tokens[:len(trunc_tokens)] = trunc_tokens
        elif self._pad_mode == 'pre':
            fixed_tokens[-len(trunc_tokens):] = trunc_tokens
        else:
            raise ValueError('{} is not a vaild '
                             'pad mode.'.format(self._pad_mode))

        return fixed_tokens.tolist()
