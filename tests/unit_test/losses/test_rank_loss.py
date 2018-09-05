import pytest

import six
import numpy as np
from keras import backend as K
from keras.layers import *
from keras.losses import mean_squared_error
from keras.losses import mean_absolute_error
from keras.utils.generic_utils import custom_object_scope

from matchzoo.losses import mz_specialized_losses
from matchzoo.losses import rank_hinge_loss
from matchzoo.losses import rank_crossentropy_loss
from matchzoo.losses import serialize
from matchzoo.losses import deserialize
from matchzoo.losses import get


def test_mz_specialized_losses():
    assert 'rank_hinge_loss' in mz_specialized_losses
    assert 'rank_crossentropy_loss' in mz_specialized_losses


def test_rank_hinge_loss():
    true_value = K.variable(np.array([[1.2], [1],
                                      [1], [1]]))
    pred_value = K.variable(np.array([[1.2], [0.1],
                                      [0], [-0.3]]))
    # no kwargs
    expected_loss = (0+1-0.3+0)/2.0
    loss = K.eval(rank_hinge_loss()(true_value, pred_value))
    assert np.isclose(expected_loss, loss)
    # with kwargs and margin in it
    expected_loss = (2+0.1-1.2+2-0.3+0) / 2.0
    loss = K.eval(rank_hinge_loss(kwargs={'margin': 2.})(true_value, pred_value))
    assert np.isclose(expected_loss, loss)
    # with kwargs and margin not in it
    expected_loss = (1-0.3+0) / 2.0
    loss = K.eval(rank_hinge_loss(kwargs={'marginal': 2.})(true_value, pred_value))
    assert np.isclose(expected_loss, loss)


def test_rank_crossentropy_loss():
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)
    true_value = K.variable(np.array([[1.], [0.],
                                      [0.], [1.]]))
    pred_value = K.variable(np.array([[0.8], [0.1],
                                      [0.8], [0.1]]))
    # no kwargs
    expected_loss = (-np.log(softmax([0.8, 0.1])[0])-np.log(softmax([0.8, 0.1])[1]))/2
    loss = K.eval(rank_crossentropy_loss()(true_value, pred_value))
    assert np.isclose(expected_loss, loss)
    # with kwargs and neg_num not in it
    expected_loss = (-np.log(softmax([0.8, 0.1])[0])-np.log(softmax([0.8, 0.1])[1]))/2
    loss = K.eval(rank_crossentropy_loss()(true_value, pred_value))
    assert np.isclose(expected_loss, loss)
    # with kwargs and neg_num in it
    true_value = K.variable(np.array([[1.], [0.], [0.],
                                      [0.], [1.], [0.]]))
    pred_value = K.variable(np.array([[0.8], [0.1], [0.1],
                                      [0.8], [0.1], [0.1]]))
    # no kwargs
    expected_loss = (-np.log(softmax([0.8, 0.1, 0.1])[0])-np.log(softmax([0.8, 0.1, 0.1])[1]))/2
    loss = K.eval(rank_crossentropy_loss(kwargs={'neg_num': 2})(true_value, pred_value))
    assert np.isclose(expected_loss, loss)


def test_serializing_loss():
    loss = rank_hinge_loss()
    assert serialize(loss) == '_margin_loss'
    deserialized = deserialize('rank_hinge_loss')
    assert isinstance(deserialized, type(rank_hinge_loss))


def test_get():
    # case None
    identifier = None
    assert get(identifier) is None
    # case string
    identifier = 'rank_hinge_loss'
    assert isinstance(identifier, six.string_types)
    assert type(get(identifier)) == type(rank_hinge_loss)
    # case callable
    identifier = rank_hinge_loss
    assert type(get(identifier)) == type(rank_hinge_loss)
    # value error
    with pytest.raises(ValueError):
        identifier = 1
        assert type(get(identifier)) == type(rank_hinge_loss)
