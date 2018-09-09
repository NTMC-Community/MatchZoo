import pytest

import numpy as np
from keras import backend as K
from matchzoo.losses import SliceTensor


def test_shapes():
    value = np.random.random((2, 3, 4, 5))
    x = K.variable(value)
    output = SliceTensor(1, 4, 1)(x)
    assert K.eval(output).shape == (2, 3, 1, 5)

    value = np.random.random((1, 4, 2, 6))
    x = K.variable(value)
    output = SliceTensor(0, 2, 1)(x)
    assert K.eval(output).shape == (1, 2, 2, 6)


def test_value():
    value = np.random.random((2, 3, 4, 5))
    x = K.variable(value)
    output = SliceTensor(1, 2, 0)(x)
    assert np.allclose(K.eval(output), value[:, :, ::2, :])

    value = np.random.random((1, 4, 2, 6))
    x = K.variable(value)
    output = SliceTensor(-1, 3, 1)(x)
    assert np.allclose(value[:, :, :, 1::3], K.eval(output))

    value = np.random.random((1, 4, 2, 6))
    x = K.variable(value)
    output = SliceTensor(0, 4, 1)(x)
    assert np.allclose(value[:, 1:2, :, :], K.eval(output))