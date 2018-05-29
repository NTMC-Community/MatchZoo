import pytest
from matchzoo.engine.base_model import BaseModel


def test_base_model_abstract_instantiation():
    with pytest.raises(TypeError):
        model = BaseModel()
        assert model


def test_base_model_concrete_instantiation():
    class MyBaseModel(BaseModel):
        def build(self):
            pass

    model = MyBaseModel()
    assert model
