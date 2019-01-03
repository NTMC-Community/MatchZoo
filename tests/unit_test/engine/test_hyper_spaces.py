import pytest
import hyperopt.pyll.base

from matchzoo import engine


@pytest.mark.parametrize('api, kwargs', [
    (engine.hyper_spaces.choice, dict(options=[0, 1])),
    (engine.hyper_spaces.uniform, dict(low=0, high=10)),
    (engine.hyper_spaces.quniform, dict(low=0, high=10, q=2))
])
def test_choice(api, kwargs):
    hyperopt_space = api(**kwargs).convert('dummpy_label')
    assert isinstance(hyperopt_space, hyperopt.pyll.base.Apply)
