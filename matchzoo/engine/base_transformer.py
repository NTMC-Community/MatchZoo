"""Contains the base Transformer class for  all transformers."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


class TransformerMiXin(object):
    """MiXin class for all transformers in MatchZoo.

    # Methods:
        fit_transform(X, y=None, **fit_params)

    """

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
        if y is None:
            # fit method of arity 1 (unsuperised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2(supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)
