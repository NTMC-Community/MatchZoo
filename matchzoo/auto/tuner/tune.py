import typing

import matchzoo as mz
from matchzoo.engine.base_metric import BaseMetric
from .tuner import Tuner


def tune(
    params: 'mz.ParamTable',
    train_data: typing.Union[mz.DataPack, mz.DataGenerator],
    test_data: typing.Union[mz.DataPack, mz.DataGenerator],
    fit_kwargs: dict = None,
    evaluate_kwargs: dict = None,
    metric: typing.Union[str, BaseMetric] = None,
    mode: str = 'maximize',
    num_runs: int = 10,
    callbacks: typing.List['mz.auto.tuner.callbacks.Callback'] = None,
    verbose=1
):
    """
    Tune model hyper-parameters.

    A simple shorthand for using :class:`matchzoo.auto.Tuner`.

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
        >>> results = mz.auto.tune(
        ...     params=model.params,
        ...     train_data=train,
        ...     test_data=dev,
        ...     num_runs=1,
        ...     verbose=0
        ... )
        >>> sorted(results['best'].keys())
        ['#', 'params', 'sample', 'score']

    """

    tuner = Tuner(
        params=params,
        train_data=train_data,
        test_data=test_data,
        fit_kwargs=fit_kwargs,
        evaluate_kwargs=evaluate_kwargs,
        metric=metric,
        mode=mode,
        num_runs=num_runs,
        callbacks=callbacks,
        verbose=verbose
    )
    return tuner.tune()
