"""Contains the base Transformer class for  all transformers."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import abc
import pickle

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

    @abc.abstractmethod
    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization."""

    @abc.abstractmethod
    def build_processor(self):
        """Return callable that handles preprocessing."""

    @abc.abstractmethod
    def build_vocabulary(self, X, fixed_vocab=False):
        """Create wordID vector for each text in X, and vocabulary where
        fixed_vocab = False
        """

    def fit(self, X, y=None):
        """Learn the vocabulary information (global term statistical).

        Parameters
        =========
        X : list [(d1, d2, text1, text2), ...] or [(text1, text2), ...]
            a list of text pairs
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
        """Save the model."""

def load_transformer(filepath):
    """
    Load a transformer, The reverse function of :method:`BaseTransformer.save`.

    """
    if not os.path.exists(filepath):
        raise ValueError('File `%s` does not exists!' % filepath)

    params = pickle.load(open(filepath, 'rb'))
    return params['transformer_class'](params=params)

