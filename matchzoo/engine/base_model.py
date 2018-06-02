import abc
import typing
import pickle
from pathlib import Path

import keras

from . import BaseModelParams, list_available_tasks


class BaseModel(abc.ABC):
    BACKEND_FILENAME = 'backend.h5'
    PARAMS_FILENAME = 'params.pkl'
    MODEL_CLASS_FILENAME = 'class.pkl'

    def __init__(self,
                 params: typing.Optional[BaseModelParams] = None,
                 backend: keras.models.Model = None):
        self._params = params or self.get_default_params()
        self._backend = backend

    @classmethod
    def get_default_params(cls) -> BaseModelParams:
        return BaseModelParams()

    @property
    def params(self) -> BaseModelParams:
        return self._params

    @property
    def backend(self) -> keras.models.Model:
        return self._backend

    @abc.abstractmethod
    def build(self):
        """"""

    def compile(self):
        self._backend.compile(optimizer=self._params['optimizer'],
                              loss=self._params['loss'],
                              metrics=self._params['metrics'])

    def fit(self, x, y, batch_size=128, epochs=1) -> keras.callbacks.History:
        return self._backend.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def evaluate(self, x, y, batch_size=128):
        return self._backend.evaluate(x=x, y=y, batch_size=batch_size)

    def predict(self, x, batch_size=128):
        return self._backend.predict(x=x, batch_size=batch_size)

    def save(self, dirpath: typing.Union[str, Path]):
        dirpath = Path(dirpath)

        if not dirpath.exists():
            dirpath.mkdir()

        backend_path = dirpath.joinpath(BaseModel.BACKEND_FILENAME)
        self._backend.save(backend_path)

        params_path = dirpath.joinpath(BaseModel.PARAMS_FILENAME)
        pickle.dump(self._params, open(params_path, mode='wb'))

        model_class_path = dirpath.joinpath(BaseModel.MODEL_CLASS_FILENAME)
        pickle.dump(self.__class__, open(model_class_path, mode='wb'))

    def guess_and_fill_missing_params(self):
        if self._params['name'] is None:
            self._params['name'] = self.__class__.__name__

        if self._params['task'] is None:
            # index 0 points to an abstract task class
            self._params['task'] = list_available_tasks()[1]

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

        print(self._params)

    def all_params_filled(self):
        return all(value is not None for value in self._params.values())


def load_model(dirpath: typing.Union[str, Path]) -> BaseModel:
    dirpath = Path(dirpath)

    backend_path = dirpath.joinpath(BaseModel.BACKEND_FILENAME)
    backend = keras.models.load_model(backend_path)

    params_path = dirpath.joinpath(BaseModel.PARAMS_FILENAME)
    params = pickle.load(open(params_path, 'rb'))

    model_class_path = dirpath.joinpath(BaseModel.MODEL_CLASS_FILENAME)
    model_class = pickle.load(open(model_class_path, 'rb'))
    return model_class(params=params, backend=backend)
