import pytest

from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.hyper_spaces import quniform


@pytest.fixture
def param_table():
    params = ParamTable()
    params.add(Param('ham', 'Parma Ham'))
    return params


def test_get(param_table):
    assert param_table['ham'] == 'Parma Ham'


def test_set(param_table):
    new_param = Param('egg', 'Over Easy')
    param_table.set('egg', new_param)
    assert 'egg' in param_table.keys()


def test_keys(param_table):
    assert 'ham' in param_table.keys()


def test_hyper_space(param_table):
    new_param = Param(
        name='my_param',
        value=1,
        hyper_space=quniform(low=1, high=5)
    )
    param_table.add(new_param)
    hyper_space = param_table.hyper_space
    assert hyper_space
