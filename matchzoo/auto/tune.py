"""Tuner class. Currently a minimum working demo."""

import hyperopt

from matchzoo import engine
from matchzoo import models


def tune(
    model: engine.BaseModel,
    train_pack,
    test_pack,
    task,
    context=None,
    max_evals: int = 32,
    verbose=1
) -> list:
    """
    Tune a model.

    Construct a hyper parameter searching space by extracting all parameters
    in `model` that have a defined hyper space. Then, using `hyperopt` API,
    iteratively sample parameters and test for loss, and pick the best trial
    out of all. Currently a minimum working demo.

    :param model: Model to tune.
    :param train_pack: :class:`matchzoo.DataPack` to train the model.
    :param test_pack: :class:`matchzoo.DataPack` to test the model.
    :param task: :class:`matchzoo.engine.BaseTask` to execute.
    :param context: Extra information for tunning. Different for different
        models.
    :param max_evals: Number of evaluations of a single tuning process.
    :param verbose: Verbosity.

    :return: A list of trials of the tuning process.
    """

    def _test_wrapper(space):
        for key, value in space.items():
            model.params[key] = value

        if isinstance(model, models.DSSMModel):
            input_shapes = context['input_shapes']
            model.params['input_shapes'] = input_shapes

        if 'with_embedding' in model.params:
            model.params['embedding_input_dim'] = context['vocab_size']

        model.params['task'] = task
        model.guess_and_fill_missing_params(verbose=verbose)
        model.build()
        model.compile()

        model.fit(*train_pack.unpack(), verbose=verbose)
        metrics = model.evaluate(*test_pack.unpack(), verbose=verbose)

        return {
            'loss': metrics['loss'],
            'space': space,
            'status': hyperopt.STATUS_OK,
            'model_params': model.params
        }

    trials = hyperopt.Trials()
    hyper_space = model.params.hyper_space
    if not hyper_space:
        raise ValueError("Cannot auto-tune on an empty hyper space.")
    hyperopt.fmin(
        fn=_test_wrapper,
        space=hyper_space,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    return [_clean_up_trial(trial) for trial in trials]


def _clean_up_trial(trial):
    return {
        'model_params': trial['result']['model_params'],
        'sampled_params': trial['result']['space'],
        'loss': trial['result']['loss']
    }
