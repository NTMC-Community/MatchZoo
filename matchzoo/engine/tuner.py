import hyperopt
import random

from matchzoo import engine


class Tuner(object):
    """

    Example:
        >>> from matchzoo.models import DenseBaselineModel
        >>> model = DenseBaselineModel()
        >>> max_evals = 4
        >>> tuner = Tuner(model, max_evals=max_evals)
        >>> trials = tuner.tune()
        >>> len(trials) == max_evals
        True

    """
    def __init__(self, model: engine.BaseModel, max_evals=32):
        self._model = model
        self._max_evals = max_evals

    def tune(self):

        def test_wrapper(space):
            for key, value in space.items():
                self._model.params[key] = value
            self._model.guess_and_fill_missing_params()
            self._model.build()
            return {'loss'  : random.random(), 'space': space,
                    'status': hyperopt.STATUS_OK}

        trials = hyperopt.Trials()
        hyperopt.fmin(
                fn=test_wrapper,
                space=self._model.params.hyper_space,
                algo=hyperopt.tpe.suggest,
                max_evals=self._max_evals,
                trials=trials
        )
        return trials.trials
