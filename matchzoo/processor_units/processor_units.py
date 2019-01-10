"""Matchzoo toolkit for text pre-processing."""

import re
import abc
import typing
import collections

import nltk
import numpy as np

match_punc = re.compile(r'[^\w\s]')


def list_available():
    """List all available units."""
    return ProcessorUnit.__subclasses__()


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

    def __init__(self, lang: str = 'english'):
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
        >>> rv = triletter.transform(['hello', 'word'])
        >>> len(rv)
        9
        >>> rv
        ['#he', 'hel', 'ell', 'llo', 'lo#', '#wo', 'wor', 'ord', 'rd#']
        >>> triletter = NgramLetterUnit(reduce_dim=False)
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

    def transform(self, tokens: list) -> list:
        """
        Transform token into tri-letter.

        For example, `word` should be represented as `#wo`,
        `wor`, `ord` and `rd#`.

        :param tokens: list of tokens to be transformed.

        :return n_letters: generated n_letters.
        """
        n_letters = []
        for token in tokens:
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


class FrequencyFilterUnit(StatefulProcessorUnit):
    """
    Frequency filter unit.

    :param low: Lower bound, inclusive.
    :param high: Upper bound, exclusive.
    :param mode: One of `tf` (term frequency), `df` (document frequency),
        and `idf` (inverse document frequency).

    Examples::
        >>> import matchzoo as mz

    To filter based on term frequency (tf):
        >>> tf_filter = mz.processor_units.FrequencyFilterUnit(
        ...     low=2, mode='tf')
        >>> tf_filter.fit([['A', 'B', 'B'], ['C', 'C', 'C']])
        >>> tf_filter.transform(['A', 'B', 'C'])
        ['B', 'C']

    To filter based on document frequency (df):
        >>> tf_filter = mz.processor_units.FrequencyFilterUnit(
        ...     low=2, mode='df')
        >>> tf_filter.fit([['A', 'B'], ['B', 'C']])
        >>> tf_filter.transform(['A', 'B', 'C'])
        ['B']

    To filter based on inverse document frequency (idf):
        >>> idf_filter = mz.processor_units.FrequencyFilterUnit(
        ...     low=1.2, mode='idf')
        >>> idf_filter.fit([['A', 'B'], ['B', 'C', 'D']])
        >>> idf_filter.transform(['A', 'B', 'C'])
        ['A', 'C']

    """

    def __init__(self, low=0, high=float('inf'), mode='df'):
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
            raise ValueError('Mode must be one of `tf`, `df`, and `idf`.')

        for k, v in stats.items():
            if self._low <= v < self._high:
                valid_terms.add(k)

        self._state[self._mode] = valid_terms

    def transform(self, tokens: list) -> list:
        """Transform a list of tokens by filtering out unwanted words."""
        valid_terms = self._state[self._mode]
        return list(filter(lambda token: token in valid_terms, tokens))

    @classmethod
    def _tf(cls, list_of_tokens):
        stats = collections.Counter()
        for tokens in list_of_tokens:
            stats.update(tokens)
        return stats

    @classmethod
    def _df(cls, list_of_tokens):
        stats = collections.Counter()
        for tokens in list_of_tokens:
            stats.update(set(tokens))
        return stats

    @classmethod
    def _idf(cls, list_of_tokens):
        num_docs = len(list_of_tokens)
        stats = cls._df(list_of_tokens)
        for key, val in stats.most_common():
            stats[key] = np.log((1 + num_docs) / (1 + val)) + 1
        return stats


class WordHashingUnit(ProcessorUnit):
    """
    Word-hashing layer for DSSM-based models.

    The input of :class:`WordHashingUnit` should be a list of word
    sub-letter list extracted from one document. The output of is
    the word-hashing representation of this document.

    :class:`NgramLetterUnit` and :class:`VocabularyUnit` are two
    essential prerequisite of :class:`WordHashingUnit`.

    Examples:
       >>> letters = [['#te', 'tes','est', 'st#'], ['oov']]
       >>> word_hashing = WordHashingUnit(
       ...     term_index={'': 0,'st#': 1, '#te': 2, 'est': 3, 'tes': 4})
       >>> hashing = word_hashing.transform(letters)
       >>> hashing[0]
       array([0., 1., 1., 1., 1., 0.])
       >>> hashing[1]
       array([1., 0., 0., 0., 0., 0.])
       >>> hashing.shape
       (2, 6)

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

    def transform(self, terms: list) -> np.ndarray:
        """
        Transform list of :attr:`letters` into word hashing layer.

        :param terms: list of `tri_letters` generated by
            :class:`NgramLetterUnit`.
        :return: Word hashing representation of `tri-letters`.
        """
        if any(isinstance(elem, list) for elem in terms):
            # The input shape for CDSSM is
            # [[word1 ngram, ngram], [word2, ngram, ngram], ...].
            hashing = np.zeros((len(terms), len(self._term_index) + 1))
            for idx, word in enumerate(terms):
                counted_letters = collections.Counter(word)
                for key, value in counted_letters.items():
                    letter_id = self._term_index.get(key, 0)
                    hashing[idx, letter_id] = value
        else:
            # The input shape for DSSM model [ngram, ngram, ...].
            hashing = np.zeros((len(self._term_index) + 1))
            counted_letters = collections.Counter(terms)
            for key, value in counted_letters.items():
                letter_id = self._term_index.get(key, 0)
                hashing[letter_id] = value

        return hashing


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

    def __init__(self, text_length: int, pad_value: int = 0,
                 pad_mode: str = 'pre', truncate_mode: str = 'pre'):
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
        # padding process can not handle empty list as input
        if len(tokens) == 0:
            tokens = [self._pad_value]
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


class MatchingHistogramUnit(ProcessorUnit):
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
        >>> histogram = MatchingHistogramUnit(3, embedding_matrix, True, 'CH')
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

    def transform(self, text_pair: list) -> list:
        """Transform the input text."""
        text_left, text_right = text_pair
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
