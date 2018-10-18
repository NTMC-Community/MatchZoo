from matchzoo.preprocessor.cdssm_preprocessor import CDSSMPreprocessor
import pytest
import numpy as np


@pytest.fixture
def train_inputs():
    return [("id0", "id1", "how are glacier caves formed ?", "The ice facade is approximately 60 m high", 0),
            ("id0", "id2", "how are glacier caves formed ?",
             "Ice formations in the Titlis glacier cave", 0),
            ("id0", "id3", "how are glacier caves formed ?", "A glacier cave is a cave formed within the ice of a glacier", 1)]


@pytest.fixture
def validation_inputs():
    return [("id0", "id4", "how are glacier caves formed ?", "A partly submerged glacier cave on Perito Moreno Glacier .")]


def test_cdssm_preprocessor(train_inputs, validation_inputs):
    cdssm_preprocessor = CDSSMPreprocessor()
    rv_train = cdssm_preprocessor.fit_transform(
        train_inputs,
        stage='train')
    assert len(rv_train.left) == 1
    assert len(rv_train.right) == 3
    assert len(rv_train.relation) == 3
    value = rv_train.left.at['id0', 'text_left']
    assert np.array(value).shape == (10, 66)
    rv_valid = cdssm_preprocessor.fit_transform(
        validation_inputs,
        stage='test')
    assert len(rv_valid.left) == 1
    assert len(rv_valid.right) == 1
    assert len(rv_valid.relation) == 1
