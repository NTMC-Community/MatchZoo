import dill
import pytest
import keras

import matchzoo


def test_make_keras_optimizer_picklable():
    adam = keras.optimizers.Adam(lr=0.1)
    with pytest.raises(Exception):
        s = dill.dumps(adam)
        assert dill.loads(s)

    matchzoo.utils.make_keras_optimizer_picklable()

    s = dill.dumps(adam)
    assert dill.loads(s)
