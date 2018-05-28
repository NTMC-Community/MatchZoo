import os
import pytest
import tempfile
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, RepeatVector, TimeDistributed
from keras import optimizers
from keras import losses
from keras import metrics
from keras.models import save_model
from matchzoo.engine.base_model import BaseModel

@pytest.fixture
def base_model():
	model = BaseModel()
	return model

def test_init():
	model = BaseModel(a=1,b=2)
	assert model._user_given_parameters.get('a') == 1

def test_name(base_model):
	assert base_model.name == 'BaseModel'

def test_task_type(base_model):
	assert base_model.task_type is None
	base_model.task_type = 'ranking'
	assert base_model.task_type == 'ranking'

def test_trainable(base_model):
	assert base_model.trainable is True

def test_fixed_hyper_parameters(base_model):
	assert base_model.fixed_hyper_parameters == {}
	base_model.fixed_hyper_parameters = {'a': 1, 'b': 2}
	assert base_model.fixed_hyper_parameters == {'a': 1, 'b': 2}
	assert len(base_model._list_fixed_hyper_parameters) == 2 

def test_default_hyper_parameters(base_model):
	assert 'learning_rate' in base_model.default_hyper_parameters
	base_model.default_hyper_parameters = {'learning_rate': 0.05}
	assert base_model.default_hyper_parameters.get('learning_rate') == 0.05
	assert pytest.raises(ValueError)

def test_model_specific_hyper_parameters(base_model):
	assert base_model.model_specific_hyper_parameters == {}
	base_model.model_specific_hyper_parameters = {'key': 'value'}
	assert base_model.model_specific_hyper_parameters.get('key') == 'value'

def test_user_given_parameters(base_model):
	assert base_model.user_given_parameters == {}

def test_aggregate_hyper_parameters(base_model):
	base_model.model_specific_hyper_parameters = {'dim_tri_letter': 100}
	config = base_model._aggregate_hyper_parameters()
	assert isinstance(config, dict)
	assert config.get('dim_tri_letter') == 100

def test_build(base_model):
	assert base_model._build() is None

def test_compile(base_model):
	assert base_model.compile() is None

def test_train(base_model):
	assert base_model.train('a', 'b', 'c') is None

def test_load(base_model):
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(RepeatVector(3))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy],
                  sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    base_model.load(fname)
    base_model.load(fname, custom_loss={'fake_loss': losses.MSE})
    os.remove(fname)




