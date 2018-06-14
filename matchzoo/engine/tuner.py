"""Tuner class. Currently a minimum working demo."""

import hyperopt
import random

from matchzoo import engine


class Tuner(object):
    """
    Hyper parameter tuner.

    Construct a hyper parameter searching space by extracting all parameters
    in `model` that have a defined hyper space. Then, using `hyperopt` API,
    iteratively sample parameters and test for loss, and pick the best trial
    out of all.

    Currently a minimum working demo.

    Example:

        >>> from matchzoo.models import DenseBaselineModel
        >>> model = DenseBaselineModel()
        >>> max_evals = 4
        >>> tuner = Tuner(model)
        >>> trials = tuner.tune(max_evals)
        >>> len(trials) == max_evals
        True

    """

    def __init__(self, model: engine.BaseModel):
        """
        Tuner constructor.

        :param model: Model to tune.
        """
        self._model = model

    def tune(self, max_evals: int = 32) -> list:
        """
        Tune the binded model `max_evals` times.

        :param max_evals: Number of evaluations of a single tuning process.
        :return: A list of trials of the tuning process.
        """
        trials = hyperopt.Trials()
        hyperopt.fmin(
                fn=self._test_wrapper,
                space=self._model.params.hyper_space,
                algo=hyperopt.tpe.suggest,
                max_evals=max_evals,
                trials=trials
        )
        return trials.trials

    def _test_wrapper(self, space):
        for key, value in space.items():
            self._model.params[key] = value
        self._model.guess_and_fill_missing_params()
        self._model.build()
        # TODO: use model fit loss instead of a random loss
        return {'loss':   random.random(), 'space': space,
                'status': hyperopt.STATUS_OK}
