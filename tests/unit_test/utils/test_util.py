import pytest

from matchzoo import utils

def test_dotdict():
    data = {'a':0, 'b':1}
    data = utils.dotdict(data)
    assert data.a == 0
    assert data.b == 1
