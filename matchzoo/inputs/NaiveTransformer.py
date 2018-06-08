"""Contains the base Transformer class for  all transformers."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numbers
import copy

from matchzoo import engine


class NaiveTransformer(engine.BaseTransformer):
    """The simple transformer for MatchZoo.

    Examples:
        >>> X = [("This is the matchzoo toolkit.", "Here is matchzoo.")]
        >>> transformer = NaiveTransformer()
        >>> transformer.params['max_df'] = 0.999
        >>> transformer.params['min_df'] = 0.001
        >>> transformer.params['max_vocab_size'] = 30000
        >>> transformer.params['analyzer'] = 'word'
        >>> transformer.params['stop_words'] = set(['is', 'are'])
        >>> transformer.params['vocabulary'] = {'a':0, 'b':1}
        >>> transformer.params['fixed_vocab'] = False
        >>> X = transformer.fit_transform(X)

    """

    def __init__(self, max_df=1.0, min_df=0, max_vocab_size=None,
                 vocabulary=None, analyzer='word', stop_words=set(),
                 fixed_vocab=False):
        """Initialization."""
        super(NaiveTransformer, self).__init__()
        self._params['name'] = self.__class__.__name__
        self._params['transformer_class'] = self.__class__

        self._params['max_df'] = max_df
        self._params['min_df'] = min_df
        self._params['max_vocab_size'] = max_vocab_size
        self._params['analyzer'] = analyzer
        self._params['stop_words'] = stop_words
        self._params['vocabulary'] = vocabulary
        self._params['fixed_vocab'] = fixed_vocab

    def build_vocabulary(self, X, fixed_vocab=False):
        """Create the vocabulary as well as mapped X.

        If fixed_vocab is False, then generate the vocabulary.
        """
        if fixed_vocab:
            vocabulary = self._params['vocabulary']
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        new_X = copy.deepcopy(X)
        for idx, pair in enumerate(X):
            # id1 = instance['id1']
            # id2 = instance['id2']
            pair_feature = ([], [])
            for tid, text in enumerate(pair):
                for feature in analyze(text):
                    try:
                        feature_idx = vocabulary[feature]
                        pair_feature[tid].append(feature_idx)
                    except KeyError:
                        # Ignore out-of-vocabulary items for fixed_vocab=True
                        continue
            new_X[idx] = pair_feature

        if not fixed_vocab:
            # disable defaultdic behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words.")
        return vocabulary, new_X

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

    def fit_transform(self, X, y=None):
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
        if not isinstance(X, list):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "list object received.")

        self._validate_vocabulary()

        vocabulary, X = self.build_vocabulary(X, self._params['fixed_vocab'])
        if not self._params['fixed_vocab']:
            max_df = self._params['max_df']
            min_df = self._params['min_df']
            max_vocab_size = self._params['max_vocab_size']
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else max_df * self.num_docs)
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * self.num_docs)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    " max_df corresponds to < documents than min_df.")
            X, stop_words = self._limit_vocabulary(X,
                                                   vocabulary,
                                                   max_doc_count,
                                                   min_doc_count,
                                                   max_vocab_size)

            self._params['stop_words'] = stop_words
            self._params['vocabulary'] = vocabulary
        return X

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
        if not isinstance(X, list):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "list object received.")

        if not hasattr(self, 'vocabulary_'):
            self._validate_vocabulary()

        self._check_vocabulary()

        _, X = self.build_vocabulary(X, fixed_vocab=True)

        return X
