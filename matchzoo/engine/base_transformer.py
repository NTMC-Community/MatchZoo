"""Contains the base Transformer class for  all transformers."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import abc
import typing
import pickle
import re
import six
from collections import Mapping

from matchzoo import engine


class BaseTransformer(abc.ABC):
    """Base data transformer for MatchZoo.

    # Methods:
    ==========
    fit_transform(X, y=None, **fit_params)

    fit(X, y=None)

    transform(X, copy=True)

    """

    _white_spaces = re.compile(r"\s\s+")

    def __init__(self, params: typing.Optional[engine.TransformerParams]
                 = None):
        """:class:`BaseTransformer` constructor."""
        self._params = params or self.get_default_params()

    @classmethod
    def get_default_params(cls) -> engine.TransformerParams:
        """Return the default parameters for transformer."""
        return engine.TransformerParams()

    def guess_and_fill_missing_params(self):
        """
        Guess and fill missing parameters in :attribute:`params`.

        Note: likely to be moved to a higher level API in the future.
        """
        if self._params['name'] is None:
            self._params['name'] = self.__class__.__name__

        if self._params['transformer_class'] is None:
            self._params['transformer_class'] = self.__class__

    @property
    def params(self):
        """:return: model parameters."""
        return self._params

    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams."""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self._params['ngram_range']
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def _char_ngrams(self, text_document):
        """Tokenize text_document into a sequence of character n-grams."""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        min_n, max_n = self._params['ngram_range']
        if min_n == 1:
            # no need to do any slicing for unigrams
            # iterate through the string
            ngrams = list(text_document)
            min_n += 1
        else:
            ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                ngrams_append(text_document[i: i + n])
        return ngrams

    def _char_wb_ngrams(self, text_document):
        """Whitespace sensitive char-n-gram tokenization.

        Tokenize text_document into a sequence of character n-grams
            operating only inside word boundaries. n-grams at the edges
            of words are padded with space.
        """
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        min_n, max_n = self._params['ngram_range']
        ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for w in text_document.split():
            w = ' ' + w + ' '
            w_len = len(w)
            for n in range(min_n, max_n + 1):
                offset = 0
                ngrams_append(w[offset:offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams_append(w[offset:offset + n])
                if offset == 0:   # count a short word (w_len < n) only once
                    break
        return ngrams

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens."""
        if self._params['tokenizer'] is not None:
            return self._params['tokenizer']
        token_pattern = re.compile(self._params['token_pattern'])
        return lambda doc: token_pattern.findall(doc)

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization."""
        analyzer = self._params['analyzer']
        if callable(analyzer):
            return analyzer

        preprocess = self.build_preprocessor()

        if analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))

        elif analyzer == 'char_wb':
            return lambda doc: self._char_wb_ngrams(
                preprocess(doc))

        elif analyzer == 'word':
            tokenize = self.build_tokenizer()
            stop_words = self._params['stop_words']

            return lambda doc: self._word_ngrams(
                tokenize(preprocess(doc)), stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             analyzer)

    def build_preprocessor(self):
        """Return callable that handles preprocessing."""
        return lambda x: x

    def _validate_vocabulary(self):
        """Validate the vocabulary.

        Make sure the word index in correspoding with the dict indice.
        """
        vocabulary = self._params['vocabulary']
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i, t in enumerate(vocabulary):
                    if vocab.setdefault(t, i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(six.itervalues(vocabulary))
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in range(len(vocabulary)):
                    if i not in indices:
                        msg = ("Vocabulary of size %d doesn't cotain index "
                               "%d." % (len(vocabulary), i)
                               )
                        raise ValueError(msg)
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            self._params['fixed_vocab'] = True
            self._params['vocabulary'] = dict(vocabulary)
        else:
            self._params['fixed_vocab'] = False

    def _check_vocabulary(self):
        """Check if vocabulary is empty or missing."""
        if self._params['vocabulary'] is None:
            raise ValueError("Vocabulary haven't built yet.")

        if len(self._params['vocabulary']) == 0:
            raise ValueError("Vocabulary is empty.")

    @abc.abstractmethod
    def build_vocabulary(self, X):
        """Create vocabulary and map X.

        If fixed_vocab is False, then create the vocabulary.
        """

    def fit(self, X, y=None):
        """Learn the vocabulary information.

        Parameters
        =========
        X : list [(d1, d2, text1, text2), ...] or [(text1, text2), ...]
            a list of text pairs.

        """
        self.fit_transform(X)
        return self

    @abc.abstractmethod
    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transform to X and y with optional parameters fit_params.
        Then, returns a transformed version of X.

        Parameters
        -----------
        X : list [(id1, id2, text1, text2), ...] or [(text1, text2), ...]
            a list of text pairs

        y : numpy array of shape [label, ...]
            Target values.

        Returns
        -------
        X_new : list [(id1, id2, feature1, feature2), ...]
            Transformed array.

        """

    @abc.abstractmethod
    def transform(self, X, copy=True):
        """Transform dict of string vectors to a indice vectors.

        Parameters
        ==========
        X : list [(id1, id2, text1, text2), ...] or [(text1, text2), ...]
            a list of text pairs

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Return
        =======
        X_new : list [(id1, id2, feature1, feature2), ...]
            Transformed array.

        """

    def save(self, dirpath):
        """Save the transformer.

        Save the _params into the given `dirpath` by `pickle`.

        :param dirpath: file path of the saved transformer
        """
        pickle.dump(self._params, open(dirpath, mode='wb'))


def load_transformer(filepath):
    """Load a transformer, the reverse function of `BaseTransformer.save`."""
    if not os.path.exists(filepath):
        raise ValueError('File `%s` does not exists!' % filepath)

    params = pickle.load(open(filepath, 'rb'))
    return params['transformer_class'](params=params)
