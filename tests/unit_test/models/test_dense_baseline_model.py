import shutil
from pathlib import Path

from matchzoo.models import DenseBaselineModel


def test_denes_baseline_model():
    params = DenseBaselineModel.get_default_params()
    params.num_dense_units = 2048
    model = DenseBaselineModel(params=params)
    model.build()
    assert model.backend

    save_dir = 'tmp_save_dir'
    assert not Path(save_dir).exists()
    model.save(save_dir)
    assert Path(save_dir).exists()
    shutil.rmtree(save_dir)
    assert not Path(save_dir).exists()
