import copy
import typing
import logging

import hyperopt

import matchzoo as mz
from matchzoo.engine.base_metric import BaseMetric
from .callbacks.callback import Callback


class Tuner(object):
    """
   Model hyper-parameters tuner.

    `model.params.hyper_space` reprensents the model's hyper-parameters
    search space, which is the cross-product of individual hyper parameter's
    hyper space. When a `Tuner` builds a model, for each hyper parameter in
    `model.params`, if the hyper-parameter has a hyper-space, then a sample
    will be taken in the space. However, if the hyper-parameter does not
    have a hyper-space, then the default value of the hyper-parameter will
    be used.

    See `tutorials/model_tuning.ipynb` for a detailed walkthrough on usage.

    :param params: A completed parameter table to tune. Usually `model.params`
        of the desired model to tune. `params.completed()` should be `True`.
    :param train_data: Training data to use. Either a preprocessed `DataPack`,
        or a `DataGenerator`.
    :param test_data: Testing data to use. A preprocessed `DataPack`.
    :param fit_kwargs: Extra keyword arguments to pass to `fit`.
        (default: `dict(epochs=10, verbose=0)`)
    :param evaluate_kwargs: Extra keyword arguments to pass to `evaluate`.
    :param metric: Metric to tune upon. Must be one of the metrics in
        `model.params['task'].metrics`. (default: the first metric in
        `params.['task'].metrics`.
    :param mode: Either `maximize` the metric or `minimize` the metric.
        (default: 'maximize')
    :param num_runs: Number of runs. Each run takes a sample in
        `params.hyper_space` and build a model based on the sample.
        (default: 10)
    :param callbacks: A list of callbacks to handle. Handled sequentially
        at every callback point.
    :param verbose: Verbosity. (default: 1)

    Example:
        >>> import matchzoo as mz
        >>> train = mz.datasets.toy.load_data('train')
        >>> dev = mz.datasets.toy.load_data('dev')
        >>> prpr = mz.models.DenseBaseline.get_default_preprocessor()
        >>> train = prpr.fit_transform(train, verbose=0)
        >>> dev = prpr.transform(dev, verbose=0)
        >>> model = mz.models.DenseBaseline()
        >>> model.params['input_shapes'] = prpr.context['input_shapes']
        >>> model.params['task'] = mz.tasks.Ranking()
        >>> tuner = mz.auto.Tuner(
        ...     params=model.params,
        ...     train_data=train,
        ...     test_data=dev,
        ...     num_runs=1,
        ...     verbose=0
        ... )
        >>> results = tuner.tune()
        >>> sorted(results['best'].keys())
        ['#', 'params', 'sample', 'score']

    """

    def __init__(
        self,
        params: 'mz.ParamTable',
        train_data: typing.Union[mz.DataPack, mz.DataGenerator],
        test_data: typing.Union[mz.DataPack, mz.DataGenerator],
        fit_kwargs: dict = None,
        evaluate_kwargs: dict = None,
        metric: typing.Union[str, BaseMetric] = None,
        mode: str = 'maximize',
        num_runs: int = 10,
        callbacks: typing.List[Callback] = None,
        verbose=1
    ):
        """Tuner."""
        if fit_kwargs is None:
            fit_kwargs = dict(epochs=10, verbose=0)
        if evaluate_kwargs is None:
            evaluate_kwargs = {}
        if callbacks is None:
            callbacks = []

        self._validate_params(params)
        metric = metric or params['task'].metrics[0]
        self._validate_data(train_data)
        self._validate_data(test_data)
        self._validate_kwargs(fit_kwargs)
        self._validate_kwargs(evaluate_kwargs)
        self._validate_mode(mode)
        self._validate_metric(params, metric)
        self._validate_callbacks(callbacks)

        self.__curr_run_num = 0

        # these variables should not change within the same `tune` call
        self._params = params
        self._train_data = train_data
        self._test_data = test_data
        self._fit_kwargs = fit_kwargs
        self._evaluate_kwargs = evaluate_kwargs
        self._metric = metric
        self._mode = mode
        self._num_runs = num_runs
        self._callbacks = callbacks
        self._verbose = verbose

    def tune(self):
        """
        Start tuning.

        Notice that `tune` does not affect the tuner's inner state, so each
        new call to `tune` starts fresh. In other words, hyperspaces are
        suggestive only within the same `tune` call.
        """
        if self.__curr_run_num != 0:
            print(
                """WARNING: `tune` does not affect the tuner's inner state, so
                each new call to `tune` starts fresh. In other words,
                hyperspaces are suggestive only within the same `tune` call."""
            )
        self.__curr_run_num = 0
        logging.getLogger('hyperopt').setLevel(logging.CRITICAL)

        trials = hyperopt.Trials()
        hyperopt.fmin(
            fn=self._run,
            space=self._params.hyper_space,
            algo=hyperopt.tpe.suggest,
            max_evals=self._num_runs,
            trials=trials
        )

        return {
            'best': trials.best_trial['result']['mz_result'],
            'trials': [trial['result']['mz_result'] for trial in trials.trials]
        }

    def _run(self, sample):
        self.__curr_run_num += 1

        # build start
        self._handle_callbacks_run_start(sample)

        # build model
        params = self._create_full_params(sample)
        model = params['model_class'](params=params)
        model.build()
        model.compile()
        self._handle_callbacks_build_end(model)

        # fit & evaluate
        self._fit_model(model)
        lookup = self._evaluate_model(model)
        score = lookup[self._metric]

        # collect result
        # this result is for users, visible outside
        mz_result = {
            '#': self.__curr_run_num,
            'params': params,
            'sample': sample,
            'score': score
        }

        self._handle_callbacks_run_end(model, mz_result)

        if self._verbose:
            self._log_result(mz_result)

        return {
            # these two items are for hyperopt
            'loss': self._fix_loss_sign(score),
            'status': hyperopt.STATUS_OK,

            # this item is for storing matchzoo information
            'mz_result': mz_result
        }

    def _create_full_params(self, sample):
        params = copy.deepcopy(self._params)
        params.update(sample)
        return params

    def _handle_callbacks_run_start(self, sample):
        for callback in self._callbacks:
            callback.on_run_start(self, sample)

    def _handle_callbacks_build_end(self, model):
        for callback in self._callbacks:
            callback.on_build_end(self, model)

    def _handle_callbacks_run_end(self, model, result):
        for callback in self._callbacks:
            callback.on_run_end(self, model, result)

    def _fit_model(self, model):
        if isinstance(self._train_data, mz.DataPack):
            x, y = self._train_data.unpack()
            model.fit(x, y, **self._fit_kwargs)
        elif isinstance(self._train_data, mz.DataGenerator):
            model.fit_generator(self._train_data, **self._fit_kwargs)
        else:
            raise ValueError(f"Invalid data type: `train_data`."
                             f"{type(self._train_data)} received."
                             f"Must be one of `DataPack` and `DataGenerator`.")

    def _evaluate_model(self, model):
        if isinstance(self._test_data, mz.DataPack):
            x, y = self._test_data.unpack()
            return model.evaluate(x, y, **self._evaluate_kwargs)
        elif isinstance(self._test_data, mz.DataGenerator):
            return model.evaluate_generator(self._test_data,
                                            **self._evaluate_kwargs)
        else:
            raise ValueError(f"Invalid data type: `test_data`."
                             f"{type(self._test_data)} received."
                             f"Must be one of `DataPack` and `DataGenerator`.")

    def _fix_loss_sign(self, loss):
        if self._mode == 'maximize':
            loss = -loss
        return loss

    @classmethod
    def _log_result(cls, result):
        print(f"Run #{result['#']}")
        print(f"Score: {result['score']}")
        print(result['params'])
        print()

    @property
    def params(self):
        """`params` getter."""
        return self._params

    @params.setter
    def params(self, value):
        """`params` setter."""
        self._validate_params(value)
        self._validate_metric(value, self._metric)
        self._params = value

    @property
    def train_data(self):
        """`train_data` getter."""
        return self._train_data

    @train_data.setter
    def train_data(self, value):
        """`train_data` setter."""
        self._validate_data(value)
        self._train_data = value

    @property
    def test_data(self):
        """`test_data` getter."""
        return self._test_data

    @test_data.setter
    def test_data(self, value):
        """`test_data` setter."""
        self._validate_data(value)
        self._test_data = value

    @property
    def fit_kwargs(self):
        """`fit_kwargs` getter."""
        return self._fit_kwargs

    @fit_kwargs.setter
    def fit_kwargs(self, value):
        """`fit_kwargs` setter."""
        self._validate_kwargs(value)
        self._fit_kwargs = value

    @property
    def evaluate_kwargs(self):
        """`evaluate_kwargs` getter."""
        return self._evaluate_kwargs

    @evaluate_kwargs.setter
    def evaluate_kwargs(self, value):
        """`evaluate_kwargs` setter."""
        self._validate_kwargs(value)
        self._evaluate_kwargs = value

    @property
    def metric(self):
        """`metric` getter."""
        return self._metric

    @metric.setter
    def metric(self, value):
        """`metric` setter."""
        self._validate_metric(self._params, value)
        self._metric = value

    @property
    def mode(self):
        """`mode` getter."""
        return self._mode

    @mode.setter
    def mode(self, value):
        """`mode` setter."""
        self._validate_mode(value)
        self._mode = value

    @property
    def num_runs(self):
        """`num_runs` getter."""
        return self._num_runs

    @num_runs.setter
    def num_runs(self, value):
        """`num_runs` setter."""
        self._validate_num_runs(value)
        self._num_runs = value

    @property
    def callbacks(self):
        """`callbacks` getter."""
        return self._callbacks

    @callbacks.setter
    def callbacks(self, value):
        """`callbacks` setter."""
        self._validate_callbacks(value)
        self._callbacks = value

    @property
    def verbose(self):
        """`verbose` getter."""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """`verbose` setter."""
        self._verbose = value

    @classmethod
    def _validate_params(cls, params):
        if not isinstance(params, mz.ParamTable):
            raise TypeError("Only accepts a `ParamTable` instance.")
        if not params.hyper_space:
            raise ValueError("Parameter hyper-space empty.")
        if not params.completed():
            raise ValueError("Parameters not complete.")

    @classmethod
    def _validate_data(cls, train_data):
        if not isinstance(train_data, (mz.DataPack, mz.DataGenerator)):
            raise TypeError(
                "Only accepts a `DataPack` or `DataGenerator` instance.")

    @classmethod
    def _validate_kwargs(cls, kwargs):
        if not isinstance(kwargs, dict):
            raise TypeError('Only accepts a `dict` instance.')

    @classmethod
    def _validate_mode(cls, mode):
        if mode not in ('maximize', 'minimize'):
            raise ValueError('`mode` should be one of `maximize`, `minimize`.')

    @classmethod
    def _validate_metric(cls, params, metric):
        if metric not in params['task'].metrics:
            raise ValueError('Target metric does not exist in the task.')

    @classmethod
    def _validate_num_runs(cls, num_runs):
        if not isinstance(num_runs, int):
            raise TypeError('Only accepts an `int` value.')

    @classmethod
    def _validate_callbacks(cls, callbacks):
        for callback in callbacks:
            if not isinstance(callback, Callback):
                if issubclass(callback, Callback):
                    raise TypeError("Make sure to instantiate the callbacks.")
                raise TypeError('Only accepts a `callbacks` instance.')
