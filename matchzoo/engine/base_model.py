import abc
import attrdict

import keras


class Task(object):
    """"""


class Ranking(Task):
    """"""


class Classification(Task):
    """"""


class ModelConfig(attrdict.AttrDict):
    def __init__(self):
        super().__init__(
                name='default_model_name',
                text_1_max_len=30,
                text_2_max_len=30,
                trainable=False,
                task=Ranking,
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['acc']
        )


class BaseModel(abc.ABC):
    def __init__(self, config=None):
        self._config = config or self.get_default_config
        self._backend = None

        self._build()

    @classmethod
    def get_default_config(cls) -> ModelConfig:
        return ModelConfig()

    @abc.abstractmethod
    def _build(self) -> keras.models.Model:
        """"""

    @property
    def backend(self):
        return self._backend

    def compile(self):
        self._backend.compile(optimizer=self._config.optimizer, loss=self._config.loss, metrics=['acc'])

    def fit(self, text_1, text_2, labels, batch_size=128, epochs=1, **kwargs) -> keras.callbacks.History:
        return self._backend.fit(x=[text_1, text_2], y=labels, batch_size=batch_size, epochs=epochs, **kwargs)

    def evaluate(self, text_1, text_2, labels, batch_size=128, **kwargs):
        return self._backend.evaluate(x=[text_1, text_2], y=labels, batch_size=batch_size, **kwargs)

    def predict(self, text_1, text_2, batch_size=128):
        return self._backend.predict(x=[text_1, text_2], batch_size=batch_size)


class DenseBaselineModel(BaseModel):
    """
    Examples:

        >>> from matchzoo.engine.base_model import DenseBaselineModel
        >>> config = DenseBaselineModel.get_default_config()
        >>> config.num_dense_units = 1024
        >>> model = DenseBaselineModel(config)
        >>> [layer.name for layer in model.backend.layers]
        ['input_1', 'input_2', 'concatenate_1', 'dense_1', 'dense_2']
    """

    def __init__(self, config=None):
        super().__init__(config)

    @classmethod
    def get_default_config(cls):
        config = ModelConfig()
        config.num_dense_units = 512
        return config

    def _build(self):
        inputs = [keras.layers.Input((self._config.text_1_max_len,)),
                  keras.layers.Input((self._config.text_2_max_len,))]

        x = keras.layers.concatenate(inputs)
        x = keras.layers.Dense(self._config.num_dense_units, activation='relu')(x)
        x_out = keras.layers.Dense(1, activation='sigmoid')(x)

        self._backend = keras.models.Model(inputs=inputs, outputs=x_out)
