import pytest
import hyperopt.pyll.base

from matchzoo.engine import hyper_spaces


@pytest.fixture(scope='module', params=[
    lambda x: x + 2,
    lambda x: x - 2,
    lambda x: x * 2,
    lambda x: x / 2,
    lambda x: x // 2,
    lambda x: x ** 2,
    lambda x: 2 + x,
    lambda x: 2 - x,
    lambda x: 2 * x,
    lambda x: 2 / x,
    lambda x: 2 // x,
    lambda x: 2 ** x,
    lambda x: -x
])
def op(request):
    return request.param


@pytest.fixture(scope='module', params=[
    hyper_spaces.choice(options=[0, 1]),
    hyper_spaces.uniform(low=0, high=10),
    hyper_spaces.quniform(low=0, high=10, q=2)
])
def proxy(request):
    return request.param


def test_init(proxy):
    assert isinstance(proxy.convert('label'), hyperopt.pyll.base.Apply)


def test_op(proxy, op):
    assert isinstance(op(proxy).convert('label'), hyperopt.pyll.base.Apply)


def test_str(proxy):
    assert isinstance(str(proxy), str)
