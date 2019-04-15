"""Base Model."""

import abc
import typing
from pathlib import Path

import dill
import numpy as np
import keras
import keras.backend as K
import pandas as pd

import matchzoo
from matchzoo import DataGenerator
from matchzoo.engine import hyper_spaces
from matchzoo.engine.base_preprocessor import BasePreprocessor
from matchzoo.engine.base_metric import BaseMetric
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo import tasks


class BaseModel(abc.ABC):
    """
    Abstract base class of all MatchZoo models.

    MatchZoo models are wrapped over keras models, and the actual keras model
    built can be accessed by `model.backend`. `params` is a set of model
    hyper-parameters that deterministically builds a model. In other words,
    `params['model_class'](params=params)` of the same `params` always create
    models with the same structure.

    :param params: Model hyper-parameters. (default: return value from
        :meth:`get_default_params`)
    :param backend: A keras model as the model backend. Usually not passed as
        an argument.

    Example:
        >>> BaseModel()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: Can't instantiate abstract class BaseModel ...
        >>> class MyModel(BaseModel):
        ...     def build(self):
        ...         pass
        >>> isinstance(MyModel(), BaseModel)
        True

    """

    BACKEND_WEIGHTS_FILENAME = 'backend_weights.h5'
    PARAMS_FILENAME = 'params.dill'

    def __init__(
        self,
        params: typing.Optional[ParamTable] = None,
        backend: typing.Optional[keras.models.Model] = None
    ):
        """Init."""
        self._params = params or self.get_default_params()
        self._backend = backend

    @classmethod
    def get_default_params(
        cls,
        with_embedding=False,
        with_multi_layer_perceptron=False
    ) -> ParamTable:
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
            ...         params = ParamTable()
            ...         params.add(Param('num_eggs', 512))
            ...         params.add(Param('ham_type', 'Parma Ham'))
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
        params = ParamTable()
        params.add(Param(
            name='model_class', value=cls,
            desc="Model class. Used internally for save/load. "
                 "Changing this may cause unexpected behaviors."
        ))
        params.add(Param(
            name='input_shapes',
            desc="Dependent on the model and data. Should be set manually."
        ))
        params.add(Param(
            name='task',
            desc="Decides model output shape, loss, and metrics."
        ))
        params.add(Param(
            name='optimizer', value='adam',
        ))
        if with_embedding:
            params.add(Param(
                name='with_embedding', value=True,
                desc="A flag used help `auto` module. Shouldn't be changed."
            ))
            params.add(Param(
                name='embedding_input_dim',
                desc='Usually equals vocab size + 1. Should be set manually.'
            ))
            params.add(Param(
                name='embedding_output_dim',
                desc='Should be set manually.'
            ))
            params.add(Param(
                name='embedding_trainable', value=True,
                desc='`True` to enable embedding layer training, '
                     '`False` to freeze embedding parameters.'
            ))
        if with_multi_layer_perceptron:
            params.add(Param(
                name='with_multi_layer_perceptron', value=True,
                desc="A flag of whether a multiple layer perceptron is used. "
                     "Shouldn't be changed."
            ))
            params.add(Param(
                name='mlp_num_units', value=128,
                desc="Number of units in first `mlp_num_layers` layers.",
                hyper_space=hyper_spaces.quniform(8, 256, 8)
            ))
            params.add(Param(
                name='mlp_num_layers', value=3,
                desc="Number of layers of the multiple layer percetron.",
                hyper_space=hyper_spaces.quniform(1, 6)
            ))
            params.add(Param(
                name='mlp_num_fan_out', value=64,
                desc="Number of units of the layer that connects the multiple "
                     "layer percetron and the output.",
                hyper_space=hyper_spaces.quniform(4, 128, 4)
            ))
            params.add(Param(
                name='mlp_activation_func', value='relu',
                desc='Activation function used in the multiple '
                     'layer perceptron.'
            ))
        return params

    @classmethod
    def get_default_preprocessor(cls) -> BasePreprocessor:
        """
        Model default preprocessor.

        The preprocessor's transform should produce a correctly shaped data
        pack that can be used for training. Some extra configuration (e.g.
        setting `input_shapes` in :class:`matchzoo.models.DSSMModel` may be
        required on the user's end.

        :return: Default preprocessor.
        """
        return matchzoo.preprocessors.BasicPreprocessor()

    @property
    def params(self) -> ParamTable:
        """:return: model parameters."""
        return self._params

    @params.setter
    def params(self, val):
        self._params = val

    @property
    def backend(self) -> keras.models.Model:
        """:return model backend, a keras model instance."""
        if not self._backend:
            raise ValueError("Backend not found."
                             "Please build the model first.")
        else:
            return self._backend

    @abc.abstractmethod
    def build(self):
        """Build model, each subclass need to impelemnt this method."""

    def compile(self):
        """
        Compile model for training.

        Only `keras` native metrics are compiled together with backend.
        MatchZoo metrics are evaluated only through :meth:`evaluate`.
        Notice that `keras` count `loss` as one of the metrics while MatchZoo
        :class:`matchzoo.engine.BaseTask` does not.

        Examples:
            >>> from matchzoo import models
            >>> model = models.Naive()
            >>> model.guess_and_fill_missing_params(verbose=0)
            >>> model.params['task'].metrics = ['mse', 'map']
            >>> model.params['task'].metrics
            ['mse', mean_average_precision(0.0)]
            >>> model.build()
            >>> model.compile()

        """
        self._backend.compile(optimizer=self._params['optimizer'],
                              loss=self._params['task'].loss)

    def fit(
        self,
        x: typing.Union[np.ndarray, typing.List[np.ndarray], dict],
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
        generator: matchzoo.DataGenerator,
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
        x: typing.Dict[str, np.ndarray],
        y: np.ndarray,
        batch_size: int = 128
    ) -> typing.Dict[BaseMetric, float]:
        """
        Evaluate the model.

        :param x: Input data.
        :param y: Labels.
        :param batch_size: Number of samples when `predict` for evaluation.
            (default: 128)

        Examples::
            >>> import matchzoo as mz
            >>> data_pack = mz.datasets.toy.load_data()
            >>> preprocessor = mz.preprocessors.NaivePreprocessor()
            >>> data_pack = preprocessor.fit_transform(data_pack, verbose=0)
            >>> m = mz.models.DenseBaseline()
            >>> m.params['task'] = mz.tasks.Ranking()
            >>> m.params['task'].metrics = [
            ...     'acc', 'mse', 'mae', 'ce',
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
            >>> evals = m.evaluate(x, y)
            >>> type(evals)
            <class 'dict'>

        """
        result = dict()
        matchzoo_metrics, keras_metrics = self._separate_metrics()
        y_pred = self.predict(x, batch_size)

        for metric in keras_metrics:
            metric_func = keras.metrics.get(metric)
            result[metric] = K.eval(K.mean(
                metric_func(K.variable(y), K.variable(y_pred))))

        if matchzoo_metrics:
            if not isinstance(self.params['task'], tasks.Ranking):
                raise ValueError("Matchzoo metrics only works on ranking.")
            for metric in matchzoo_metrics:
                result[metric] = self._eval_metric_on_data_frame(
                    metric, x['id_left'], y, y_pred)

        return result

    def evaluate_generator(
        self,
        generator: DataGenerator,
        batch_size: int = 128
    ) -> typing.Dict['BaseMetric', float]:
        """
        Evaluate the model.

        :param generator: DataGenerator to evluate.
        :param batch_size: Batch size. (default: 128)
        """
        x, y = generator[:]
        return self.evaluate(x, y, batch_size=batch_size)

    def _separate_metrics(self):
        matchzoo_metrics = []
        keras_metrics = []
        for metric in self._params['task'].metrics:
            if isinstance(metric, BaseMetric):
                matchzoo_metrics.append(metric)
            else:
                keras_metrics.append(metric)
        return matchzoo_metrics, keras_metrics

    @classmethod
    def _eval_metric_on_data_frame(
        cls,
        metric: BaseMetric,
        id_left: typing.Union[list, np.array],
        y: typing.Union[list, np.array],
        y_pred: typing.Union[list, np.array]
    ):
        eval_df = pd.DataFrame(data={
            'id': id_left,
            'true': y.squeeze(),
            'pred': y_pred.squeeze()
        })
        assert isinstance(metric, BaseMetric)
        val = eval_df.groupby(by='id').apply(
            lambda df: metric(df['true'].values, df['pred'].values)
        ).mean()
        return val

    def predict(
        self,
        x: typing.Dict[str, np.ndarray],
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

        Example:

            >>> import matchzoo as mz
            >>> model = mz.models.Naive()
            >>> model.guess_and_fill_missing_params(verbose=0)
            >>> model.build()
            >>> model.save('temp-model')
            >>> import shutil
            >>> shutil.rmtree('temp-model')

        """
        dirpath = Path(dirpath)
        params_path = dirpath.joinpath(self.PARAMS_FILENAME)
        weights_path = dirpath.joinpath(self.BACKEND_WEIGHTS_FILENAME)

        if not dirpath.exists():
            dirpath.mkdir(parents=True)
        else:
            raise FileExistsError(f'{dirpath} already exist, fail to save.')

        self._backend.save_weights(weights_path)
        with open(params_path, mode='wb') as params_file:
            dill.dump(self._params, params_file)

    def get_embedding_layer(
        self, name: str = 'embedding'
    ) -> keras.layers.Layer:
        """
        Get the embedding layer.

        All MatchZoo models with a single embedding layer set the embedding
        layer name to `embedding`, and this method should return that layer.

        :param name: Name of the embedding layer. (default: `embedding`)
        """
        for layer in self._backend.layers:
            if layer.name == name:
                return layer
        raise ValueError(f"Layer {name} not found. Initialize your embedding "
                         f"layer with `name='{name}'`.")

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
        self.get_embedding_layer(name).set_weights([embedding_matrix])

    def guess_and_fill_missing_params(self, verbose=1):
        """
        Guess and fill missing parameters in :attr:`params`.

        Use this method to automatically fill-in other hyper parameters.
        This involves some guessing so the parameter it fills could be
        wrong. For example, the default task is `Ranking`, and if we do not
        set it to `Classification` manaully for data packs prepared for
        classification, then the shape of the model output and the data will
        mismatch.

        :param verbose: Verbosity.
        """
        self._params.get('task').set_default(tasks.Ranking(), verbose)
        self._params.get('input_shapes').set_default([(30,), (30,)], verbose)
        if 'with_embedding' in self._params:
            self._params.get('embedding_input_dim').set_default(300, verbose)
            self._params.get('embedding_output_dim').set_default(300, verbose)

    def _set_param_default(self, name: str,
                           default_val: str, verbose: int = 0):
        if self._params[name] is None:
            self._params[name] = default_val
            if verbose:
                print(f"Parameter \"{name}\" set to {default_val}.")

    def _make_inputs(self) -> list:
        input_left = keras.layers.Input(
            name='text_left',
            shape=self._params['input_shapes'][0]
        )
        input_right = keras.layers.Input(
            name='text_right',
            shape=self._params['input_shapes'][1]
        )
        return [input_left, input_right]

    def _make_output_layer(self) -> keras.layers.Layer:
        """:return: a correctly shaped keras dense layer for model output."""
        task = self._params['task']
        if isinstance(task, tasks.Classification):
            return keras.layers.Dense(task.num_classes, activation='softmax')
        elif isinstance(task, tasks.Ranking):
            return keras.layers.Dense(1, activation='linear')
        else:
            raise ValueError(f"{task} is not a valid task type."
                             f"Must be in `Ranking` and `Classification`.")

    def _make_embedding_layer(
        self,
        name: str = 'embedding',
        **kwargs
    ) -> keras.layers.Layer:
        return keras.layers.Embedding(
            self._params['embedding_input_dim'],
            self._params['embedding_output_dim'],
            trainable=self._params['embedding_trainable'],
            name=name,
            **kwargs
        )

    def _make_multi_layer_perceptron_layer(self) -> keras.layers.Layer:
        # TODO: do not create new layers for a second call
        if not self._params['with_multi_layer_perceptron']:
            raise AttributeError(
                'Parameter `with_multi_layer_perception` not set.')

        def _wrapper(x):
            activation = self._params['mlp_activation_func']
            for _ in range(self._params['mlp_num_layers']):
                x = keras.layers.Dense(self._params['mlp_num_units'],
                                       activation=activation)(x)
            return keras.layers.Dense(self._params['mlp_num_fan_out'],
                                      activation=activation)(x)

        return _wrapper


def load_model(dirpath: typing.Union[str, Path]) -> BaseModel:
    """
    Load a model. The reverse function of :meth:`BaseModel.save`.

    :param dirpath: directory path of the saved model
    :return: a :class:`BaseModel` instance

    Example:

            >>> import matchzoo as mz
            >>> model = mz.models.Naive()
            >>> model.guess_and_fill_missing_params(verbose=0)
            >>> model.build()
            >>> model.save('my-model')
            >>> model.params.keys() == mz.load_model('my-model').params.keys()
            True
            >>> import shutil
            >>> shutil.rmtree('my-model')

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
