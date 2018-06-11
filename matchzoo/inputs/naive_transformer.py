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

    @classmethod
    def get_default_params(cls) -> engine.TransformerParams:
        """:return: model default parameters."""
        params = engine.TransformerParams()

        params['max_df'] = 1.0
        params['min_df'] = 0.0
        params['max_vocab_size'] = 2147483648  # 2**31 - 1
        params['num_of_docs'] = 1
        params['fixed_vocab'] = False
        return params

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
        stop_words = set()
        if high is None and low is None and limit is None:
            return X, stop_words
        return X, stop_words

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
                             else max_df * self._params['num_of_docs'])
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else min_df * self._params['num_of_docs'])
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
