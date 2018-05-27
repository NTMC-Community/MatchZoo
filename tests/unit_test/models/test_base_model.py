import pytest
from matchzoo.engine.base_model import BaseModel

@pytest.fixture
def base_model():
	model = BaseModel()
	return model

def test_name(base_model):
	assert base_model.name == 'BaseModel'

def test_task_type(base_model):
	assert base_model.task_type is None
	base_model.tasks_type = 'unsupported type'
	assert pytest.raises(ValueError)

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
	assert base_model.default_hyper_parameters['learning_rate'] == 0.05
	assert pytest.raises(ValueError)
