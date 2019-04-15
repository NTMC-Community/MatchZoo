from matchzoo.engine.base_model import BaseModel
from matchzoo.auto.tuner.callbacks.callback import Callback


class LambdaCallback(Callback):
    """
    LambdaCallback. Just a shorthand for creating a callback class.

    See :class:`matchzoo.tuner.callbacks.Callback` for more details.

    Example:

        >>> import matchzoo as mz
        >>> model = mz.models.Naive()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> data = mz.datasets.toy.load_data()
        >>> data = model.get_default_preprocessor().fit_transform(
        ...     data, verbose=0)
        >>> def show_inputs(*args):
        ...     print(' '.join(map(str, map(type, args))))
        >>> callback = mz.auto.tuner.callbacks.LambdaCallback(
        ...     on_run_start=show_inputs,
        ...     on_build_end=show_inputs,
        ...     on_run_end=show_inputs
        ... )
        >>> _ = mz.auto.tune(
        ...     params=model.params,
        ...     train_data=data,
        ...     test_data=data,
        ...     num_runs=1,
        ...     callbacks=[callback],
        ...     verbose=0,
        ... ) # noqa: E501
        <class 'matchzoo.auto.tuner.tuner.Tuner'> <class 'dict'>
        <class 'matchzoo.auto.tuner.tuner.Tuner'> <class 'matchzoo.models.naive.Naive'>
        <class 'matchzoo.auto.tuner.tuner.Tuner'> <class 'matchzoo.models.naive.Naive'> <class 'dict'>

    """

    def __init__(
        self,
        on_run_start=None,
        on_build_end=None,
        on_run_end=None
    ):
        """Init."""
        self._on_run_start = on_run_start
        self._on_build_end = on_build_end
        self._on_run_end = on_run_end

    def on_run_start(self, tuner, sample: dict):
        """`on_run_start`."""
        if self._on_run_start:
            self._on_run_start(tuner, sample)

    def on_build_end(self, tuner, model: BaseModel):
        """`on_build_end`."""
        if self._on_build_end:
            self._on_build_end(tuner, model)

    def on_run_end(self, tuner, model: BaseModel, result: dict):
        """`on_run_end`."""
        if self._on_run_end:
            self._on_run_end(tuner, model, result)
