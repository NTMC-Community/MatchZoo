import pytest
from matchzoo.engine.base_transformer import BaseTransformer

from collections import defaultdict
import copy

def test_transformer():
    class DummyTransformer(BaseTransformer):
        """This is a dummy class for """

        def __init__(self):
            super(DummyTransformer, self).__init__()
            self.params['name'] = 'DummyTransformer'

        def build_vocabulary(self, X):
            fixed_vocab = self._params['fixed_vocab']
            if fixed_vocab:
                vocabulary = self.params['vocabulary']
            else:
                vocabulary = defaultdict()
                vocabulary.default_factory = vocabulary.__len__

            analyze = self.build_analyzer()
            new_X = copy.deepcopy(X)
            for idx, pair in enumerate(X):
                pair_feature = ([], [])
                for tid, text in enumerate(pair):
                    for feature in analyze(text):
                        try:
                            feature_idx = vocabulary[feature]
                            pair_feature[tid].append(feature_idx)
                        except KeyError:
                            continue
                new_X[idx] = pair_feature
            if not fixed_vocab:
                vocabulary = dict(vocabulary)
            return vocabulary, new_X

        def transform(self, X):
            self._validate_vocabulary()
            self._check_vocabulary()
            _, X = self.build_vocabulary(X)
            return X

        def fit_transform(self, X, y=None):
            self._validate_vocabulary()
            vocabulary, X = self.build_vocabulary(X)
            self._params['vocabulary'] = vocabulary
            self._params['fixed_vocab'] = True
            return X

    transformer = DummyTransformer()
    x = [("Here is a test case .", "This case is to test BaseTransformer .")]
    y = [0]

    # the test
    assert transformer.params['name'] == 'DummyTransformer'

    new_x = transformer.fit_transform(x, y)
    assert new_x == [([0, 1, 2, 3, 4], [5, 4, 1, 6, 3, 7])]

    new_x = transformer.fit(x, y).transform(x)
    assert new_x == [([0, 1, 2, 3, 4], [5, 4, 1, 6, 3, 7])]
