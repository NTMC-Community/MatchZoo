"""Base Model."""

import abc
import typing
from pathlib import Path

import dill
import numpy as np
import keras

from matchzoo import engine
from matchzoo import tasks


class BaseModel(abc.ABC):
    """Abstract base class of all matchzoo models."""

    BACKEND_FILENAME = 'backend.h5'
    PARAMS_FILENAME = 'params.dill'

    def __init__(
            self,
            params: engine.ParamTable = None,
            backend: keras.models.Model = None
    ):
        """
        :class:`BaseModel` constructor.

        :param params: model parameters, if not set, return value from
            :meth:`get_default_params` will be used
        :param backend: a keras model as the model backend
        """
        self._params = params or self.get_default_params()
        self._backend = backend

    @classmethod
    def get_default_params(cls) -> engine.ParamTable:
        """
        Model default parameters.

        The common usage is to instantiate :class:`matchzoo.engine.ModelParams`
            first, then set the model specific parametrs.

        Examples:
            >>> class MyModel(BaseModel):
            ...     def build(self):
            ...         print(self._params['num_eggs'], 'eggs')
            ...         print('and', self._params['ham_type'])
            ...
            ...     @classmethod
            ...     def get_default_params(cls):
            ...         params = engine.ParamTable()
            ...         params.add(engine.Param('num_eggs', 512))
            ...         params.add(engine.Param('ham_type', 'Parma Ham'))
            ...         return params
            >>> my_model = MyModel()
            >>> my_model.build()
            512 eggs
            and Parma Ham

        Notice that all parameters must be serialisable for the entire model
        to be serialisable. Therefore, it's strongly recommended to use python
        native data types to store parameters.

        :return: model parameters

        """
        params = engine.ParamTable()
        params.add(engine.Param('name'))
        params.add(engine.Param('model_class', cls))
        params.add(engine.Param('input_shapes'))
        params.add(engine.Param('task'))
        params.add(engine.Param('metrics'))
        params.add(engine.Param('loss'))
        params.add(engine.Param('optimizer'))
        return params

    @property
    def params(self) -> engine.ParamTable:
        """:return: model parameters."""
        return self._params

    @property
    def backend(self) -> keras.models.Model:
        """:return model backend, a keras model instance."""
        return self._backend

    @abc.abstractmethod
    def build(self):
        """
        Build model, each sub class need to impelemnt this method.

        Example:

            >>> BaseModel()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            TypeError: Can't instantiate abstract class BaseModel ...
            >>> class MyModel(BaseModel):
            ...     def build(self):
            ...         pass
            >>> MyModel
            <class 'matchzoo.engine.base_model.MyModel'>
        """

    def compile(self):
        """Compile model for training."""
        self._backend.compile(optimizer=self._params['optimizer'],
                              loss=self._params['loss'],
                              metrics=self._params['metrics'])

    def fit(
            self,
            x: typing.Union[np.ndarray, typing.List[np.ndarray]],
            y: np.ndarray,
            batch_size: int = 128,
            epochs: int = 1,
            verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Fit the model.

        See :meth:`keras.models.Model.fit` for more details.

        :param x: input data.
        :param y: labels.
        :param batch_size: number of samples per gradient update.
        :param epochs: number of epochs to train the model.
        :param verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = verbose,
            2 = one log line per epoch.

        :return: A `keras.callbacks.History` instance. Its history attribute
            contains all information collected during training.
        """
        return self._backend.fit(x=x, y=y,
                                 batch_size=batch_size, epochs=epochs,
                                 verbose=verbose)

    def evaluate(
            self,
            x: typing.Union[np.ndarray, typing.List[np.ndarray]],
            y: np.ndarray,
            batch_size: int = 128,
            verbose: int = 1
    ) -> typing.Union[float, typing.List[float]]:
        """
        Evaluate the model.

        See :meth:`keras.models.Model.evaluate` for more details.

        :param x: input data
        :param y: labels
        :param batch_size: number of samples per gradient update
        :param verbose: verbosity mode, 0 or 1
        :return: scalar test loss (if the model has a single output and no
            metrics) or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.backend.metrics_names` will
            give you the display labels for the scalar outputs.

        """
        return self._backend.evaluate(x=x, y=y,
                                      batch_size=batch_size, verbose=verbose)

    def predict(
            self,
            x: typing.Union[np.ndarray, typing.List[np.ndarray]],
            batch_size=128
    ) -> np.ndarray:
        """
        Generate output predictions for the input samples.

        See :meth:`keras.models.Model.predict` for more details.

        :param x: input data
        :param batch_size: number of samples per gradient update
        :return: numpy array(s) of predictions
        """
        return self._backend.predict(x=x, batch_size=batch_size)

    def save(self, dirpath: typing.Union[str, Path]):
        """
        Save the model.

        A saved model is represented as a directory with two files. One is a
        model parameters file saved by `pickle`, and the other one is a model
        h5 file saved by `keras`.

        :param dirpath: directory path of the saved model
        """
        dirpath = Path(dirpath)

        if dirpath.exists():
            raise FileExistsError
        else:
            dirpath.mkdir()

        backend_file_path = dirpath.joinpath(self.BACKEND_FILENAME)
        self._backend.save(backend_file_path)

        params_file_path = dirpath.joinpath(self.PARAMS_FILENAME)
        dill.dump(self._params, open(params_file_path, mode='wb'))

    def guess_and_fill_missing_params(self):
        """
        Guess and fill missing parameters in :attr:`params`.

        Note: likely to be moved to a higher level API in the future.
        """
        if self._params['name'] is None:
            self._params['name'] = self.__class__.__name__

        if self._params['task'] is None:
            # index 0 points to an abstract task class
            self._params['task'] = engine.list_available_tasks()[1]()

        if self._params['input_shapes'] is None:
            self._params['input_shapes'] = [(30,), (30,)]

        if self._params['metrics'] is None:
            task = self._params['task']
            available_metrics = task.list_available_metrics()
            self._params['metrics'] = available_metrics

        if self._params['loss'] is None:
            available_losses = self._params['task'].list_available_losses()
            self._params['loss'] = available_losses[0]

        if self._params['optimizer'] is None:
            self._params['optimizer'] = 'adam'

    def _make_output_layer(self):
        """:return: a correctly shaped keras dense layer for model output."""
        task = self._params['task']
        if isinstance(task, tasks.Classification):
            return keras.layers.Dense(task.num_classes, activation='softmax')
        elif isinstance(task, tasks.Ranking):
            return keras.layers.Dense(1, activation='sigmoid')
        else:
            raise ValueError("Invalid task type.")


def load_model(dirpath: typing.Union[str, Path]) -> BaseModel:
    """
    Load a model. The reverse function of :meth:`BaseModel.save`.

    :param dirpath: directory path of the saved model
    :return: a :class:`BaseModel` instance
    """
    dirpath = Path(dirpath)

    backend_file_path = dirpath.joinpath(BaseModel.BACKEND_FILENAME)
    backend = keras.models.load_model(backend_file_path)

    params_file_path = dirpath.joinpath(BaseModel.PARAMS_FILENAME)
    params = dill.load(open(params_file_path, 'rb'))

    return params['model_class'](params=params, backend=backend)
