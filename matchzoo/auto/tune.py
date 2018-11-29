"""Tuner class. Currently a minimum working demo."""

import hyperopt

from matchzoo import engine
from matchzoo import models


def tune(
    model: engine.BaseModel,
    train_pack,
    test_pack,
    task,
    max_evals: int = 32,
    context=None,
) -> list:
    """
    Tune the `model` `max_evals` times.

    Construct a hyper parameter searching space by extracting all parameters
    in `model` that have a defined hyper space. Then, using `hyperopt` API,
    iteratively sample parameters and test for loss, and pick the best trial
    out of all. Currently a minimum working demo.

    :param model:
    :param max_evals: Number of evaluations of a single tuning process.
    :return: A list of trials of the tuning process.
    """

    def _test_wrapper(space):
        for key, value in space.items():
            model.params[key] = value

        if isinstance(model, models.DSSMModel):
            input_shapes = context['input_shapes']
            model.params['input_shapes'] = input_shapes

        model.params['task'] = task
        model.guess_and_fill_missing_params()
        model.build()
        model.compile()

        model.fit(*train_pack.unpack())
        metrics = model.evaluate(*test_pack.unpack())

        return {
            'loss': metrics['loss'],
            'space': space,
            'status': hyperopt.STATUS_OK,
            'model_params': model.params
        }

    trials = hyperopt.Trials()
    hyperopt.fmin(
        fn=_test_wrapper,
        space=model.params.hyper_space,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    return [clean_up_trial(trial) for trial in trials]


def clean_up_trial(trial):
    return {
        'model_params': trial['result']['model_params'],
        'sampled_params': trial['result']['space'],
        'loss': trial['result']['loss']
    }
