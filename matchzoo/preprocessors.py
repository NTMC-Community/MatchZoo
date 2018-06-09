"""Matchzoo toolkit for text pre-processing."""

import re
import nltk
import jieba
import many_stop_words
from functools import reduce


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

    :return terms: list of tokenized terms.
    """
    return nltk.word_tokenize(raw)


def segmentation(raw: str) -> list:
    """
    Chinese sentence segmentation.

    :param raw: raw input given by user.

    :return terms: list of segmentated terms.
    """
    return list(jieba.cut(raw))


def to_lowercase(terms: list) -> list:
    """
    Convert list of terms to lower case.

    :param terms: list of terms.

    :return terms: lower-cased list of terms.
    """
    return [term.lower() for term in terms]


def remove_stopwords(terms: list, lang: str='en') -> list:
    """
    Remove stopwords from list of tokenized terms.

    :param terms: list of tokenized terms.
    :param lang: language code for stopwords.

    :return terms: list of tokenized terms without stopwords.
    """
    return [term for term in terms if term not in get_stopwords(lang)]


def remove_punctuation(terms: list) -> list:
    """
    Remove punctuations from list of terms.

    :param terms: list of 

    :return rv: terms  without punctuation.
    """
    rv = []
    for term in terms:
        term = re.sub(r'[^\w\s]', '', term)
        if term != '':
            rv.append(term)
    return rv


def remove_digits(terms: list) -> list:
    """
    Remove digits from list of terms.

    :param terms: list of terms to be filtered.

    :return terms: list of terms without digits.
    """
    return [term for term in terms if not term.isdigit()]


def stemming(terms: list) -> list:
    """
    reducing inflected words to their word stem, base or root form.

    :param text: list of string to be stemmed.
    :param mode: stemming algorithm, porter stemer by default.

    :return terms: stemmed term.
    """
    porter_stemmer = nltk.stem.PorterStemmer()
    return [porter_stemmer.stem(term) for term in terms]


def lemmatization(terms: list) -> list:
    """
    Lemmatization a sequence of terms.

    :param terms: list of terms to be lemmatized.

    :return terms: list of lemmatizd terms.
    """
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(term, pos='v') for term in terms]


def chain(*funcs):
    """
    Chain a list of functions execute as pipeline.

    Examples:
        >>> processor = chain(tokenizer,
        ...                   to_lowercase,
        ...                   remove_punctuation,
        ...                   remove_digits)
        >>> rv = processor("This is an Example sentence to BE ! cleaned with digits 31.")

    :param *funcs: list of functions to be executed.

    :return rv: return value of the last function.
    """
    return lambda x: reduce(lambda f, g: g(f), list(funcs), x)
