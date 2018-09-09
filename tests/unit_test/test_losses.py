import pytest

import numpy as np
from keras import backend as K

# from matchzoo.losses import _margin
# from matchzoo.losses import _neg_num
# from matchzoo.losses import set_margin
# from matchzoo.losses import set_neg_num
from matchzoo.losses import rank_hinge_loss
from matchzoo.losses import rank_crossentropy_loss


# def test_set_margin():
#     set_margin(2.0)
#     assert _margin == 2.0
#
#
# def test_set_neg_num():
#     set_neg_num(3)
#     assert _neg_num == 3


def test_hinge_loss():
    true_value = K.variable(np.array([[1.2], [1],
                                      [1], [1]]))
    pred_value = K.variable(np.array([[1.2], [0.1],
                                      [0], [-0.3]]))
    # no kwargs
    expected_loss = (0 + 1 - 0.3 + 0) / 2.0
    loss = K.eval(rank_hinge_loss(true_value, pred_value))
    assert np.isclose(expected_loss, loss)
    # # with kwargs and margin in it
    # expected_loss = (2 + 0.1 - 1.2 + 2 - 0.3 + 0) / 2.0
    # set_margin(2)
    # loss = K.eval(rank_hinge_loss(true_value, pred_value))
    # assert np.isclose(expected_loss, loss)


def test_rank_crossentropy_loss():
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)
    # true_value = K.variable(np.array([[1., 1.], [0., 0.],
    #                                   [0., 0.], [1., 1.]]))
    # pred_value = K.variable(np.array([[0.8, 0.8], [0.1, 0.1],
    #                                   [0.8, 0.8], [0.1, 0.1]]))
    true_value = K.variable(np.array([[1.], [0.],
                                      [0.], [1.]]))
    pred_value = K.variable(np.array([[0.8], [0.1],
                                      [0.8], [0.1]]))
    # no kwargs
    expected_loss = (-np.log(softmax([0.8, 0.1])[0])-np.log(softmax([0.8, 0.1])[1]))/2
    loss = K.eval(rank_crossentropy_loss(true_value, pred_value))
    print(loss, expected_loss)
    assert np.isclose(expected_loss, loss)
    # # with kwargs and neg_num in it
    # # true_value = K.variable(np.array([[1., 1.], [0., 0.], [0., 0.],
    # #                                   [0., 0.], [1., 1.], [0., 0.]]))
    # # pred_value = K.variable(np.array([[0.8, 0.8], [0.1, 0.1], [0.1, 0.1],
    # #                                   [0.8, 0.8], [0.1, 0.1], [0.1, 0.1]]))
    # true_value = K.variable(np.array([[1.], [0.], [0.],
    #                                   [0.], [1.], [0.]]))
    # pred_value = K.variable(np.array([[0.8], [0.1], [0.1],
    #                                   [0.8], [0.1], [0.1]]))
    # # no kwargs
    # # set_neg_num(2)
    # expected_loss = (-np.log(softmax([0.8, 0.1, 0.1])[0])-np.log(softmax([0.8, 0.1, 0.1])[1]))/2
    # loss = K.eval(rank_crossentropy_loss(true_value, pred_value))
    # print(loss, expected_loss)
    # assert np.isclose(expected_loss, loss)
