import abc
import pickle
from pathlib import Path

import keras

from engine.model_params import ModelParams

BACKEND_FILENAME = 'backend.h5'
PARAMS_FILENAME = 'params.pkl'
MODEL_CLASS_FILENAME = 'class.pkl'


def load_model(dirpath):
    backend_path = dirpath.joinpath(BACKEND_FILENAME)
    backend = keras.models.load_model(backend_path)

    params_path = dirpath.joinpath(PARAMS_FILENAME)
    params = pickle.load(params_path)

    model_class_path = dirpath.joinpath(MODEL_CLASS_FILENAME)
    model_class = pickle.load(model_class_path)
    return model_class(params=params, backend=backend)


class BaseModel(abc.ABC):
    def __init__(self, params=None, backend=None):
        self._params = params or self.get_default_params()
        self._backend = backend

    @classmethod
    def get_default_params(cls) -> ModelParams:
        return ModelParams()

    @property
    def params(self):
        return self._params

    @property
    def backend(self):
        return self._backend

    @abc.abstractmethod
    def build(self):
        """"""

    def compile(self):
        self._backend.compile(optimizer=self._params.optimizer, loss=self._params.loss, metrics=['acc'])

    def fit(self, text_1, text_2, labels, batch_size=128, epochs=1, **kwargs) -> keras.callbacks.History:
        return self._backend.fit(x=[text_1, text_2], y=labels, batch_size=batch_size, epochs=epochs, **kwargs)

    def evaluate(self, text_1, text_2, labels, batch_size=128, **kwargs):
        return self._backend.evaluate(x=[text_1, text_2], y=labels, batch_size=batch_size, **kwargs)

    def predict(self, text_1, text_2, batch_size=128):
        return self._backend.predict(x=[text_1, text_2], batch_size=batch_size)

    def save(self, dirpath):
        dirpath = Path(dirpath)
        if not dirpath.exists():
            dirpath.mkdir()

        backend_path = dirpath.joinpath(BACKEND_FILENAME)
        self._backend.save(backend_path)

        params_path = dirpath.joinpath(PARAMS_FILENAME)
        pickle.dump(self._params, open(params_path, mode='wb'))

        model_class_path = dirpath.joinpath(MODEL_CLASS_FILENAME)
        pickle.dump(self.__class__, open(model_class_path, mode='wb'))
