"""Matchzoo toolkit for text pre-processing."""

import re
import nltk
import jieba
import typing
import many_stop_words
from functools import reduce


match_punc = re.compile('[^\w\s]')


def get_stopwords(lang: str='en') -> list:
    """
    Get stopwords based on language.

    :params lang: language code.

    :return stop_list: list of stop words.
    """
    return many_stop_words.get_stop_words(lang)


def tokenizer(raw: str) -> list:
    """
    Word tokenization.

    :param raw: raw input given by user.

    :return tokens: list of tokenized tokens.
    """
    return nltk.word_tokenize(raw)


def segmentation(raw: str) -> list:
    """
    Chinese sentence segmentation.

    :param raw: raw input given by user.

    :return tokens: list of segmentated tokens.
    """
    return list(jieba.cut(raw))


def to_lowercase(tokens: list) -> list:
    """
    Convert list of tokens to lower case.

    :param tokens: list of tokens.

    :return tokens: lower-cased list of tokens.
    """
    return [token.lower() for token in tokens]


def remove_stopwords(tokens: list, lang: str='en') -> list:
    """
    Remove stopwords from list of tokenized tokens.

    :param tokens: list of tokenized tokens.
    :param lang: language code for stopwords.

    :return tokens: list of tokenized tokens without stopwords.
    """
    return [token for token in tokens if token not in get_stopwords(lang)]


def remove_punctuation(tokens: list) -> list:
    """
    Remove punctuations from list of tokens.

    :param tokens: list of

    :return rv: tokens  without punctuation.
    """
    return [token for token in tokens if not match_punc.search(token)]


def remove_digits(tokens: list) -> list:
    """
    Remove digits from list of tokens.

    :param tokens: list of tokens to be filtered.

    :return tokens: tokens of tokens without digits.
    """
    return [token for token in tokens if not token.isdigit()]


def stemming(tokens: list, stemmer: str = 'porter') -> list:
    """
    Reducing inflected words to their word stem, base or root form.

    :param tokens: list of string to be stemmed.

    :raise ValueError: stemmer type should be porter or lancaster.

    :return tokens: stemmed token.
    """
    if stemmer == 'porter':
        porter_stemmer = nltk.stem.PorterStemmer()
        return [porter_stemmer.stem(token) for token in tokens]
    elif stemmer == 'lancaster' or stemmer == 'krovetz':
        lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
        return [lancaster_stemmer.stem(token) for token in tokens]
    else:
        raise ValueError(
            'Not supported supported stemmer type: {}'.format(stemmer))


def lemmatization(tokens: list) -> list:
    """
    Lemmatization a sequence of tokens.

    :param tokens: list of tokens to be lemmatized.

    :return tokens: list of lemmatizd tokens.
    """
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token, pos='v') for token in tokens]


def chain(*funcs: typing.Callable) -> typing.Callable:
    """
    Chain a list of functions execute as pipeline.

    Examples:
        >>> processor = chain(tokenizer,
        ...                   to_lowercase,
        ...                   remove_punctuation,
        ...                   stemming,
        ...                   remove_digits)
        >>> rv = processor("An Example sentence to BE cleaned!")
        >>> import functools
        >>> processor = chain(tokenizer,
        ...                   functools.partial(stemming, stemmer='lancaster'))
        >>> rv = processor("An example sentence to BE cleaned!")

    :param *funcs: functions to be executed.

    :return rv: return value of the last function.

    """
    return lambda x: reduce(lambda f, g: g(f), list(funcs), x)
