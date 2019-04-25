import pytest
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

import matchzoo as mz


@pytest.fixture(scope='module')
def data(num_train=200, num_test=0, input_shape=(10,),
         output_shape=(2,), classification=True, num_classes=2):
    """Generates test data to train a model on.
    classification=True overrides output_shape
    (i.e. output_shape is set to (1,)) and the output
    consists in integers in [0, num_class-1].
    Otherwise: float output with shape output_shape.
    """
    samples = num_train + num_test
    if classification:
        y = np.random.randint(0, num_classes, size=(samples,))
        X = np.zeros((samples,) + input_shape)
        for i in range(samples):
            X[i] = np.random.normal(loc=y[i], scale=0.7, size=input_shape)
    else:
        y_loc = np.random.random((samples,))
        X = np.zeros((samples,) + input_shape)
        y = np.zeros((samples,) + output_shape)
        for i in range(samples):
            X[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=input_shape)
            y[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=output_shape)

    x_train, y_train = X[:num_train], y[:num_train]
    y_train = to_categorical(y_train)
    return x_train, y_train


@pytest.fixture(scope='module')
def model(data):
    x_train, y_train = data
    model = Sequential()
    model.add(Dense(10, input_shape=(x_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dense(y_train.shape[1]))
    model.add(Activation('softmax'))
    return model


@pytest.fixture(scope='module', params=[
    {'dense_1': 0.8},
])
def multipliers(request):
    return request.param


@pytest.fixture(scope='module', params=mz.optimizers.list_available())
def optimizer_class(request):
    return request.param


@pytest.fixture(scope='module')
def optimizer(optimizer_class, multipliers):
    optimizer = optimizer_class(multipliers=multipliers)
    return optimizer


def test_optimizer(data, model, optimizer, target=0.55):
    x_train, y_train = data
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, batch_size=16, verbose=0)
    assert history.history['acc'][-1] >= target

    config = keras.optimizers.serialize(optimizer)
    custom_objects = {optimizer.__class__.__name__: optimizer.__class__}
    optim = keras.optimizers.deserialize(config, custom_objects)
    new_config = keras.optimizers.serialize(optim)
    assert config == new_config
