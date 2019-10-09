import typing
from pathlib import Path
import uuid

import matchzoo as mz
from matchzoo.engine.base_model import BaseModel
from .callback import Callback


class SaveModel(Callback):
    """
    Save trained model.

    For each trained model, a UUID will be generated as the `model_id`, the
    model will be saved under the `dir_path/model_id`. A `model_id` key will
    also be inserted into the result, which will visible in the return value of
    the `tune` method.

    :param dir_path: Path to save the models to. (default:
        `matchzoo.USER_TUNED_MODELS_DIR`)

    """

    def __init__(
        self,
        dir_path: typing.Union[str, Path] = mz.USER_TUNED_MODELS_DIR
    ):
        """Init."""
        self._dir_path = dir_path

    def on_run_end(self, tuner, model: BaseModel, result: dict):
        """Save model on run end."""
        model_id = str(uuid.uuid4())
        model.save(self._dir_path.joinpath(model_id))
        result['model_id'] = model_id
