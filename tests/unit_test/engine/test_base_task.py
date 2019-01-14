import pytest
from matchzoo import engine


def test_base_task_instantiation():
    with pytest.raises(TypeError):
        engine.BaseTask()
