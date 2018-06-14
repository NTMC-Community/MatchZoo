"""Tuner class. Currently a minimum working demo."""

import hyperopt
import random

from matchzoo import engine


def tune(model: engine.BaseModel, max_evals: int = 32) -> list:
    """
    Tune the `model` `max_evals` times.

    Construct a hyper parameter searching space by extracting all parameters
    in `model` that have a defined hyper space. Then, using `hyperopt` API,
    iteratively sample parameters and test for loss, and pick the best trial
    out of all. Currently a minimum working demo.

    :param model:
    :param max_evals: Number of evaluations of a single tuning process.
    :return: A list of trials of the tuning process.

    Example:

        >>> from matchzoo.models import DenseBaselineModel
        >>> model = DenseBaselineModel()
        >>> max_evals = 4
        >>> trials = tune(model, max_evals)
        >>> len(trials) == max_evals
        True

    """
    trials = hyperopt.Trials()

    def _test_wrapper(space):
        for key, value in space.items():
            model.params[key] = value
        model.guess_and_fill_missing_params()
        model.build()
        # the random loss is for demostration purpose without actual meaning
        return {'loss':   random.random(), 'space': space,
                'status': hyperopt.STATUS_OK}

    hyperopt.fmin(
            fn=_test_wrapper,
            space=model.params.hyper_space,
            algo=hyperopt.tpe.suggest,
            max_evals=max_evals,
            trials=trials
    )
    return trials.trials
