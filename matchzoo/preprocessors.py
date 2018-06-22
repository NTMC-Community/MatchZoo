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
    def fit(self, input: typing.Any) -> dict:
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
