"""Callbacks."""
import typing
from pathlib import Path

import numpy as np
import keras

import matchzoo
from matchzoo.engine.base_model import BaseModel


class EvaluateAllMetrics(keras.callbacks.Callback):
    """
    Callback to evaluate all metrics.

    MatchZoo metrics can not be evaluated batch-wise since they require
    dataset-level information. As a result, MatchZoo metrics are not
    evaluated automatically when a Model `fit`. When this callback is used,
    all metrics, including MatchZoo metrics and Keras metrics, are evluated
    once every `once_every` epochs.

    :param model: Model to evaluate.
    :param x: X.
    :param y: y.
    :param once_every: Evaluation only triggers when `epoch % once_every == 0`.
        (default: 1, i.e. evaluate on every epoch's end)
    :param batch_size: Number of samples per evaluation. This only affects the
        evaluation of Keras metrics, since MatchZoo metrics are always
        evaluated using the full data.
    :param model_save_path: Directory path to save the model after each
        evaluate callback, (default: None, i.e., no saving.)
    :param verbose: Verbosity.
    """

    def __init__(
        self,
        model: 'BaseModel',
        x: typing.Union[np.ndarray, typing.List[np.ndarray]],
        y: np.ndarray,
        once_every: int = 1,
        batch_size: int = 128,
        model_save_path: str = None,
        verbose=1
    ):
        """Initializer."""
        super().__init__()
        self._model = model
        self._dev_x = x
        self._dev_y = y
        self._valid_steps = once_every
        self._batch_size = batch_size
        self._model_save_path = model_save_path
        self._verbose = verbose

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Called at the end of en epoch.

        :param epoch: integer, index of epoch.
        :param logs: dictionary of logs.
        :return: dictionary of logs.
        """
        if (epoch + 1) % self._valid_steps == 0:
            val_logs = self._model.evaluate(self._dev_x, self._dev_y,
                                            self._batch_size)
            if self._verbose:
                print('Validation: ' + ' - '.join(
                    f'{k}: {v}' for k, v in val_logs.items()))
            for k, v in val_logs.items():
                logs[k] = v
            if self._model_save_path:
                curr_path = self._model_save_path + str('%d/' % (epoch + 1))
                self._model.save(curr_path)
