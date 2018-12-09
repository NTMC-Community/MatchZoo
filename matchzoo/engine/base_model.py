"""Base Model."""

import abc
import typing
import logging
from pathlib import Path

import dill
import numpy as np
import keras
import pandas as pd

import matchzoo
from matchzoo import DataGenerator
from matchzoo import engine
from matchzoo import tasks

logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """Abstract base class of all matchzoo models."""

    BACKEND_WEIGHTS_FILENAME = 'backend_weights.h5'
    PARAMS_FILENAME = 'params.dill'

    class EvaluateOnCall(keras.callbacks.Callback):
        """:class:`EvaluateOncall` evaluate validation datasets on callback."""

        def __init__(self,
                     matchzoo_model: 'BaseModel',
                     x: typing.Union[np.ndarray, typing.List[np.ndarray]],
                     y: np.ndarray,
                     valid_steps=3,
                     batch_size: int = 32):
            """
            :class:`EvaluateOnCall` constructor.

            :param matchzoo_model: model to evaluate.
            :param x: input data.
            :param y: labels.
            :param valid_steps: integer, skipping steps(number of batches) to
                call the :class:`EvaluateOnCall`.
            :param batch_size: integer, number of instances in a batch.

            """
            super().__init__()
            self._mz_model = matchzoo_model
            self._dev_x = x
            self._dev_y = y
            self._valid_steps = valid_steps
            self._batch_size = batch_size

        def on_epoch_end(self, epoch, logs=None):
            """
            Called at the end of en epoch.

            :param epoch: integer, index of epoch.
            :param logs: dictionary of logs.
            :return: dictionary of logs.
            """
            if epoch % self._valid_steps == 0:
                val_logs = self._mz_model.evaluate(self._dev_x, self._dev_y,
                                                   self._batch_size, verbose=0)
                logger.info('Validation: ' + ' - '.join(
                    f'{k}:{v:f}' for k, v in val_logs.items()))

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
    def get_default_params(cls, with_embedding=False) -> engine.ParamTable:
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
        params.add(engine.Param('optimizer'))
        if with_embedding:
            params.add(engine.Param('with_embedding', True))
            params.add(engine.Param('embedding_input_dim'))
            params.add(engine.Param('embedding_output_dim'))
            params.add(engine.Param('embedding_trainable'))
        return params

    @classmethod
    def get_default_preprocessor(cls) -> engine.BasePreprocessor:
        """
        Model default preprocessor.

        The preprocessor's transform should produce a correctly shaped data
        pack that can be used for training. Some extra configuration (e.g.
        setting `input_shapes` in :class:`matchzoo.models.DSSMModel` may be
        required on the user's end.

        :return: Default preprocessor.
        """
        return matchzoo.preprocessors.NaivePreprocessor()

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
            >>> assert MyModel()
        """

    def compile(self):
        """
        Compile model for training.

        Only `keras` native metrics are compiled together with backend.
        MatchZoo metrics are evaluated only through :method:`evaluate`.
        Notice that `keras` count `loss` as one of the metrics while MatchZoo
        :class:`matchzoo.engine.BaseTask` does not.

        Examples:
            >>> from matchzoo import models
            >>> model = models.NaiveModel()
            >>> model.guess_and_fill_missing_params(verbose=0)
            >>> model.params['task'].metrics = ['mse', 'map']
            >>> model.params['task'].metrics
            ['mse', mean_average_precision(0)]
            >>> model.build()
            >>> model.compile()
            >>> model.backend.metrics_names
            ['loss', 'mean_squared_error']

        """
        _, keras_metrics = self._separate_metrics()
        self._backend.compile(optimizer=self._params['optimizer'],
                              loss=self._params['task'].loss,
                              metrics=keras_metrics)

    def fit(
        self,
        x: typing.Union[np.ndarray, typing.List[np.ndarray]],
        y: np.ndarray,
        batch_size: int = 128,
        epochs: int = 1,
        verbose: int = 1,
        **kwargs
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

        Key word arguments not listed above will be propagated to keras's fit.

        :return: A `keras.callbacks.History` instance. Its history attribute
            contains all information collected during training.
        """
        return self._backend.fit(x=x, y=y,
                                 batch_size=batch_size, epochs=epochs,
                                 verbose=verbose, **kwargs)

    def fit_generator(
        self,
        generator: DataGenerator,
        epochs: int = 1,
        verbose: int = 1,
        **kwargs
    ) -> keras.callbacks.History:
        """
        Fit the model with matchzoo `generator`.

        See :meth:`keras.models.Model.fit_generator` for more details.

        :param generator: A generator, an instance of
            :class:`engine.DataGenerator`.
        :param epochs: Number of epochs to train the model.
        :param verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = verbose,
            2 = one log line per epoch.

        :return: A `keras.callbacks.History` instance. Its history attribute
            contains all information collected during training.
        """
        return self._backend.fit_generator(
            generator=generator,
            epochs=epochs,
            verbose=verbose, **kwargs
        )

    def evaluate(
        self,
        x: typing.Union[np.ndarray, typing.List[np.ndarray],
                        typing.Dict[str, np.ndarray]],
        y: np.ndarray,
        batch_size: int = 128,
        verbose: int = 1
    ) -> typing.Dict[str, float]:
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

        Examples::
            >>> import matchzoo as mz
            >>> data_pack = mz.datasets.toy.load_train_rank_data()
            >>> preprocessor = mz.preprocessors.NaivePreprocessor()
            >>> data_pack = preprocessor.fit_transform(data_pack)
            >>> m = mz.models.DenseBaselineModel()
            >>> m.params['task'] = mz.tasks.Ranking()
            >>> m.params['task'].metrics = [
            ...     'acc', 'mse', 'mae',
            ...     'average_precision', 'precision', 'dcg', 'ndcg',
            ...     'mean_reciprocal_rank', 'mean_average_precision', 'mrr',
            ...     'map', 'MAP',
            ...     mz.metrics.AveragePrecision(threshold=1),
            ...     mz.metrics.Precision(k=2, threshold=2),
            ...     mz.metrics.DiscountedCumulativeGain(k=2),
            ...     mz.metrics.NormalizedDiscountedCumulativeGain(
            ...         k=3, threshold=-1),
            ...     mz.metrics.MeanReciprocalRank(threshold=2),
            ...     mz.metrics.MeanAveragePrecision(threshold=3)
            ... ]
            >>> m.guess_and_fill_missing_params(verbose=0)
            >>> m.build()
            >>> m.compile()
            >>> x, y = data_pack.unpack()
            >>> evals = m.evaluate(x, y, verbose=0)
            >>> type(evals)
            <class 'dict'>

        """
        result = self._evaluate_backend(x, y, batch_size, verbose)
        matchzoo_metrics, _ = self._separate_metrics()
        if matchzoo_metrics:
            df = self._build_data_frame_for_eval(x, y, batch_size)
            for metric in matchzoo_metrics:
                result[metric] = self._eval_metric_on_data_frame(metric, df)
        return result

    def _evaluate_backend(self, x, y, batch_size, verbose):
        vals = self._backend.evaluate(x=x, y=y,
                                      batch_size=batch_size,
                                      verbose=verbose)
        if not isinstance(vals, list):
            vals = [vals]
        return dict(zip(self._backend.metrics_names, vals))

    def _separate_metrics(self):
        matchzoo_metrics = []
        keras_metrics = []
        for metric in self._params['task'].metrics:
            if isinstance(metric, engine.BaseMetric):
                matchzoo_metrics.append(metric)
            else:
                keras_metrics.append(metric)
        return matchzoo_metrics, keras_metrics

    def _build_data_frame_for_eval(self, x, y, batch_size):
        y_pred = self.predict(x, batch_size).reshape((-1,))
        return pd.DataFrame(data={
            'id': x['id_left'].tolist(),
            'true': y.tolist(),
            'pred': y_pred.tolist()
        })

    @classmethod
    def _eval_metric_on_data_frame(cls, metric: engine.BaseMetric, eval_df):
        assert isinstance(metric, engine.BaseMetric)
        val = eval_df.groupby(by='id').apply(
            lambda df: metric(df['true'], df['pred'])
        ).mean()
        return val

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
        params_path = dirpath.joinpath(self.PARAMS_FILENAME)
        weights_path = dirpath.joinpath(self.BACKEND_WEIGHTS_FILENAME)

        if not dirpath.exists():
            dirpath.mkdir()
        else:
            raise FileExistsError

        self._backend.save_weights(weights_path)
        with open(params_path, mode='wb') as params_file:
            dill.dump(self._params, params_file)

    def load_embedding_matrix(
        self,
        embedding_matrix: np.ndarray,
        name: str = 'embedding'
    ):
        """
        Load an embedding matrix.

        Load an embedding matrix into the model's embedding layer. The name
        of the embedding layer is specified by `name`. For models with only
        one embedding layer, set `name='embedding'` when creating the keras
        layer, and use the default `name` when load the matrix. For models
        with more than one embedding layers, initialize keras layer with
        different layer names, and set `name` accordingly to load a matrix
        to a chosen layer.

        :param embedding_matrix: Embedding matrix to be loaded.
        :param name: Name of the layer. (default: 'embedding')
        """
        for layer in self._backend.layers:
            if layer.name == name:
                layer.set_weights([embedding_matrix])
                return
        raise ValueError(f"layer {name} not found. Initialize your embedding "
                         f"layer with `name='{name}'`.")

    def guess_and_fill_missing_params(self, verbose=1):
        """
        Guess and fill missing parameters in :attr:`params`.

        :param verbose: Verbosity.
        """
        self._params.get('name').set_default(self.__class__.__name__, verbose)
        task = engine.list_available_tasks()[1]()
        self._params.get('task').set_default(task, verbose)
        self._params.get('input_shapes').set_default([(30,), (30,)], verbose)
        self._params.get('optimizer').set_default('adam', verbose)
        if 'with_embedding' in self._params:
            self._params.get('embedding_input_dim').set_default(300, verbose)
            self._params.get('embedding_output_dim').set_default(300, verbose)
            self._params.get('embedding_trainable').set_default(True, verbose)

    def _set_param_default(self, name, default_val, verbose):
        if self._params[name] is None:
            self._params[name] = default_val
            if verbose:
                print(f"Parameter \"{name}\" set to {default_val}.")

    def _make_inputs(self):
        input_left = keras.layers.Input(
            name='text_left',
            shape=self._params['input_shapes'][0]
        )
        input_right = keras.layers.Input(
            name='text_right',
            shape=self._params['input_shapes'][1]
        )
        return [input_left, input_right]

    def _make_output_layer(self):
        """:return: a correctly shaped keras dense layer for model output."""
        task = self._params['task']
        if isinstance(task, tasks.Classification):
            return keras.layers.Dense(task.num_classes, activation='softmax')
        elif isinstance(task, tasks.Ranking):
            return keras.layers.Dense(1, activation='linear')
        else:
            raise ValueError("Invalid task type.")

    def _make_embedding_layer(self, name='embedding'):
        return keras.layers.Embedding(
            self._params['embedding_input_dim'],
            self._params['embedding_output_dim'],
            trainable=self._params['embedding_trainable'],
            name=name
        )


def load_model(dirpath: typing.Union[str, Path]) -> BaseModel:
    """
    Load a model. The reverse function of :meth:`BaseModel.save`.

    :param dirpath: directory path of the saved model
    :return: a :class:`BaseModel` instance
    """
    dirpath = Path(dirpath)

    params_path = dirpath.joinpath(BaseModel.PARAMS_FILENAME)
    weights_path = dirpath.joinpath(BaseModel.BACKEND_WEIGHTS_FILENAME)

    with open(params_path, mode='rb') as params_file:
        params = dill.load(params_file)

    model_instance = params['model_class'](params=params)
    model_instance.build()
    model_instance.compile()
    model_instance.backend.load_weights(weights_path)
    return model_instance
