import pytest

import matchzoo as mz


@pytest.fixture(scope='module')
def tuner():
    model = mz.models.DenseBaseline()
    prpr = model.get_default_preprocessor()
    train_raw = mz.datasets.toy.load_data('train')
    dev_raw = mz.datasets.toy.load_data('dev')
    prpr.fit(train_raw)
    model.params.update(prpr.context)
    model.guess_and_fill_missing_params()
    return mz.auto.Tuner(
        params=model.params,
        train_data=prpr.transform(train_raw, verbose=0),
        test_data=prpr.transform(dev_raw, verbose=0)
    )


@pytest.mark.parametrize('attr', [
    'params',
    'train_data',
    'test_data',
    'fit_kwargs',
    'evaluate_kwargs',
    'metric',
    'mode',
    'num_runs',
    'callbacks',
    'verbose'
])
def test_getters_setters(tuner, attr):
    val = getattr(tuner, attr)
    setattr(tuner, attr, val)
    assert getattr(tuner, attr) is val


def test_tuning(tuner):
    tuner.num_runs = 1
    assert tuner.tune()
