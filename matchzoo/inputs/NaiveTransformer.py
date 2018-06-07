"""Contains the base Transformer class for  all transformers."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import abc
from collections import Mapping, defaultdict
import numbers
import six

class BaseTransformer(abc.ABC):
    """Basic data transformer for MatchZoo.

    # Attributes
    ============
    vocabulary_ : dict
        A mapping of terms to feature indices.

    stop_words_ : set
        Terms that were ignored because they either:

            - occured in too many documents (`max_df`)
            - occured in too few documents (`min_df`)
            - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.

    # Methods:
    ==========
    fit_transform(X, y=None, **fit_params)

    fit(X, y=None)

    transform(X, copy=True)

    """

    def __init__(self, max_df=1.0, min_df=0, max_vocab_size=None,
                 vocabulary=None, analyzer='word'):
        self.name = 'BaseTransformer'
        self.stop_words_ = set()
        self.max_df = max_df
        self.min_df = min_df
        self.max_vocab_size = max_vocab_size
        self.vocabulary = vocabulary
        self.analyzer = analyzer

    def _validate_vocabulary(self):
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, set):
                vocabulary = sorted(vocabulary)
            if not isinstance(vocabulary, Mapping):
                vocab = {}
                for i,t in enumerate(vocabulary):
                    if vocab.setdefault(t,i) != i:
                        msg = "Duplicate term in vocabulary: %r" % t
                        raise ValueError(msg)
                vocabulary = vocab
            else:
                indices = set(six.itervalues(vocabulary))
                if len(indices) != len(vocabulary):
                    raise ValueError("Vocabulary contains repeated indices.")
                for i in xrange(len(vocabulary)):
                    if i not in indices:
                        msg = ("Vocabulary of size %d doesn't cotain index "
                                "%d." % (len(vocabulary), i))
                        raise ValueError(msg)
            if not vocabulary:
                raise ValueError("empty vocabulary passed to fit")
            self.fixed_vocabulary_ = True
            self.vocabulary_ = dict(vocabulary)
        else:
            self.fixed_vocabulary_ = False

    def _validate_params(self):
        return

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization"""
        if callable(self.analyzer):
            return self.analyzer

        #preprocess = self.build_preprocessor()

        if self.analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(doc))

        elif self.analyzer == 'char_wb':
            return lambda doc:self._char_wb_ngrams(preprocess(doc))
        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            tokenizer = self.build_tokenizer()
            return lambda doc: self._word_ngrams(tokenize(preprocess(doc)),
                                                 stop_words)
        else:
            raise ValueError('%s is not a valid tokenization scheme/analyzer' %
                             self.analyzer)


    @classmethod
    def build_vocabulary(self, X, fixed_vocab=False):
        """Create wordID vector for each text in X, and vocabulary where
        fixed_vocab = False
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        new_X = copy(X)
        for idx, instance in enumerate(X):
            # id1 = instance['id1']
            # id2 = instance['id2']
            pair_text = (instance['text1'],instance['text2'])
            pair_feature = ([], [])
            for tid, text in enumerate(pair_text):
                for feature in analyze(text):
                    try:
                        feature_idx = vocabulary[feature]
                        pair_feature[tid].append(feature_idx)
                    except KeyError:
                        # Ignore out-of-vocabulary items for fixed_vocab=True
                        continue
            new_X[idx]['text1'] = pair_feature[0]
            new_X[idx]['text2'] = pair_feature[1]

        if not fixed_vocab:
            # disable defaultdic behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulay; perhaps the documents only"
                                 " contain stop words.")

        return vocabulay, new_X


    def _limit_vocabulary(self, X, vocabulary, high=None, low=None,
                          limit=None):
        """Remove too rare or too common words.

        Prune words that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to at
        most the limit most frequent.

        This does not prune samples with zero words.
        """
        if high is None and low is None and limit is None:
            return X, set()

        # Calculate a mask based on document frequencies


    @classmethod
    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transform to X and y with optional parameters fit_params.
        Then, returns a transformed version of X.

        Parameters
        -----------
        X : numpy array of shape [n_samples, n_pairs]
            Training set.

        y : numpy array of shape [batch_size]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_pairs]
            Transformed array.

        """
        if not isinstance(X, dict):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "dict object received.")

        self._validate_vocabulary()

        vocabulary, X= self.build_vocabulary(X, self.fixed_vocabulary_)
        if not self.fixed_vocabulary_:
            max_df = self.max_df
            min_df = self.min_df
            max_vocab_size = self.max_vocab_size
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * self.num_docs)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * self.num_docs)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    " max_df corresponds to < documents than min_df.")
            X, self.stop_words_ = self._limit_vocabulary(X, vocabulary,
                                                         max_doc_count,
                                                         min_doc_count,
                                                         max_vocab_size)
            self.vocabulary_ = vocabulary
        return X

    @abc.abstractmethod
    def fit(self, X, y=None):
        """Learn the vocabulary information (global term statistical).

        Parameters
        =========
        X : list, [{'t1':string, 't2':string}]
            a list of text pairs
        """
        self.fit_transform(X)
        return self

    @abc.abstractmethod
    def transform(self, X, copy=True):
        """Transform dict of string vectors to a indice vectors.

        Parameters
        ==========
        X : list [{'d1': text1, 'd2': text2, t1':string, 't2':string}]
            a list of text pairs

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        """
        if not isinstance(X, dict):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "dict object received.")

        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self._check_vocabulary()

        _, X = self.build_vocabulary(X, fixed_vocab=True)

        return X

