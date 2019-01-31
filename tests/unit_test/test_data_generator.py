import copy

import pytest
import keras

import matchzoo as mz


@pytest.fixture(scope='module')
def data_gen():
    return mz.DataGenerator(mz.datasets.toy.load_data())


@pytest.mark.parametrize('attr', [
    'callbacks',
    'num_neg',
    'num_dup',
    'mode',
    'batch_size',
    'shuffle',

])
def test_data_generator_getters_setters(data_gen, attr):
    assert hasattr(data_gen, attr)
    val = getattr(data_gen, attr)
    setattr(data_gen, attr, val)
    assert getattr(data_gen, attr) == val


def test_resample():
    model = mz.models.Naive()
    prpr = model.get_default_preprocessor()
    data_raw = mz.datasets.toy.load_data()
    data = prpr.fit_transform(data_raw)
    model.params.update(prpr.context)
    model.params['task'] = mz.tasks.Ranking()
    model.build()
    model.compile()

    data_gen = mz.DataGenerator(
        data_pack=data,
        mode='pair',
        resample=True,
        batch_size=4
    )

    class CheckResample(keras.callbacks.Callback):
        def __init__(self, data_gen):
            super().__init__()
            self._data_gen = data_gen
            self._orig_indices = None
            self._flags = []

        def on_epoch_end(self, epoch, logs=None):
            curr_indices = self._data_gen.batch_indices
            if not self._orig_indices:
                self._orig_indices = copy.deepcopy(curr_indices)
            else:
                self._flags.append(self._orig_indices != curr_indices)
                self._orig_indices = curr_indices

    check_resample = CheckResample(data_gen)
    model.fit_generator(data_gen, epochs=5, callbacks=[check_resample])
    assert check_resample._flags
    assert all(check_resample._flags)
