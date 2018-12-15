"""An implementation of aNMM Model."""

import typing
import logging
import numpy as np

import keras
import keras.backend as K
from keras.activations import softmax

from matchzoo import engine

logger = logging.getLogger(__name__)


def show_tensor_info(name: str, input: np.ndarray):
    """Show the tensor shapes."""
    logger.info(
        '[Layer]: %s\t[Shape]: %s\n' % (name, input.get_shape().as_list()))


class ANMMModel(engine.BaseModel):
    """
    ANMM Model.

    Examples:
        >>> model = ANMMModel()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'
        params['input_shapes'] = [(5,), (300,)]
        params.add(engine.Param('bin_num', value=20,
            hyper_space=engine.hyper_spaces.quniform(low=2, high=100)
        ))
        params.add(engine.Param('hidden_sizes', [5, 1]))
        return params

    def build(self):
        """Build model structure."""
