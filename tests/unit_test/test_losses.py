import pytest

import numpy as np
from keras import backend as K

import matchzoo.losses
from matchzoo.losses import rank_hinge_loss
from matchzoo.losses import rank_crossentropy_loss


def test_hinge_loss():
    true_value = K.variable(np.array([[1.2], [1],
                                      [1], [1]]))
    pred_value = K.variable(np.array([[1.2], [0.1],
                                      [0], [-0.3]]))
    expected_loss = (0 + 1 - 0.3 + 0) / 2.0
    loss = K.eval(rank_hinge_loss(true_value, pred_value))
    assert np.isclose(expected_loss, loss)
    matchzoo.losses._margin = 2
    expected_loss = (2 + 0.1 - 1.2 + 2 - 0.3 + 0) / 2.0
    loss = K.eval(rank_hinge_loss(true_value, pred_value))
    assert np.isclose(expected_loss, loss)


def test_rank_crossentropy_loss():
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)
    true_value = K.variable(np.array([[1.], [0.],
                                      [0.], [1.]]))
    pred_value = K.variable(np.array([[0.8], [0.1],
                                      [0.8], [0.1]]))
    expected_loss = (-np.log(softmax([0.8, 0.1])[0])-np.log(softmax([0.8, 0.1])[1]))/2
    loss = K.eval(rank_crossentropy_loss(true_value, pred_value))
    assert np.isclose(expected_loss, loss)
    true_value = K.variable(np.array([[1.], [0.], [0.],
                                      [0.], [1.], [0.]]))
    pred_value = K.variable(np.array([[0.8], [0.1], [0.1],
                                      [0.8], [0.1], [0.1]]))
    matchzoo.losses._neg_num = 2
    expected_loss = (-np.log(softmax([0.8, 0.1, 0.1])[0])-np.log(softmax([0.8, 0.1, 0.1])[1]))/2
    loss = K.eval(rank_crossentropy_loss(true_value, pred_value))
    print(loss, expected_loss)
    assert np.isclose(expected_loss, loss)
