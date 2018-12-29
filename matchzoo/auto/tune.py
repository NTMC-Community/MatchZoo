"""Tuner class. Currently a minimum working demo."""

import copy
from pathlib import Path

import hyperopt

import matchzoo
from matchzoo import engine


class Tuner(object):
    def __init__(self, **kwargs):
        self._params = engine.ParamTable()

        self._params.add(engine.Param(
            'model', validator=self._validate_model
        ))
        self._params.add(engine.Param(
            'train_pack', validator=lambda x: isinstance(x, matchzoo.DataPack)
        ))
        self._params.add(engine.Param(
            'test_pack', validator=lambda x: isinstance(x, matchzoo.DataPack)
        ))
        self._params.add(engine.Param(
            'mode', validator=lambda mode: mode in ('min', 'max')
        ))

        input_model_metrics = self._params['model']['task'].metrics
        self._params.add(engine.Param(
            'optimizing_metric', 'loss',
            validator=lambda metric: metric in input_model_metrics
        ))
        self._params.add(engine.Param(
            'num_evals', 32,
            validator=lambda max_evals: isinstance(max_evals, int)
        ))
        self._params.add(engine.Param(
            'verbose', 1, validator=lambda verbose: verbose in (0, 1)
        ))
        self._params.add(engine.Param(
            'save_dir', validator=lambda path: not Path(path).exists()
        ))
        for key, value in kwargs:
            self._params[key] = value

    @classmethod
    def _validate_model(cls, model):
        if not isinstance(model, engine.BaseModel):
            return False
        elif not model.params.hyper_space:
            print("Model hyper space empty.")
            return False
        else:
            return True

    @property
    def params(self):
        return self._params

    def tune(self):
        orig_params = copy.deepcopy(self._params['model'].params)

        trials = hyperopt.Trials()

        hyperopt.fmin(
            fn=self._test_wrapper,
            space=self._params['model'].params.hyper_space,
            algo=hyperopt.tpe.suggest,
            max_evals=self._params['num_evals'],
            trials=trials
        )
        return [_clean_up_trial(trial) for trial in trials]

    def _test_wrapper(self, space):
        for key, value in space.items():
            self._params['model'].params[key] = value
        results = self._eval()

        score = results[self._params['optimizing_metric']]

        if self._params['mode'] == 'max':
            score = -score

        return {
            'loss': score,
            'space': space,
            'status': hyperopt.STATUS_OK,
            'model_params': self._params['model'].params
        }

    def _eval(self):
        model = self._params['model']
        x_train, y_test = self._params['train_pack'].unpack()
        x_test, y_test = self._params['test_pack'].unpack()
        verbose = self._params['verbose']

        model.build()
        model.compile()
        model.fit(x_train, y_test, verbose=verbose)
        return model.evaluate(x_test, y_test, verbose=verbose)


# def tune(
#     model: engine.BaseModel,
#     train_pack,
#     test_pack,
#     mode: str = 'min',
#     optimizing_metric='loss',
#     max_evals: int = 32,
#     verbose=1
# ) -> list:
#     """
#     Tune a model.
#
#     Construct a hyper parameter searching space by extracting all parameters
#     in `model` that have a defined hyper space. Then, using `hyperopt` API,
#     iteratively sample parameters and test for loss, and pick the best trial
#     out of all. Currently a minimum working demo.
#
#     :param model: Model to tune.
#     :param train_pack: :class:`matchzoo.DataPack` to train the model.
#     :param test_pack: :class:`matchzoo.DataPack` to test the model.
#     :param optimizing_metric:
#     :param mode: 'min' to minimize the metric or 'max' to maximize.
#     :param max_evals: Number of evaluations of a single tuning process.
#     :param verbose: Verbosity.
#
#     :return: A list of trials of the tuning process.
#     """
#     #
#     # def _test_wrapper(space):
#     #     for key, value in space.items():
#     #         model.params[key] = value
#     #     results = _eval_model()
#     #
#     #     score = results[optimizing_metric]
#     #
#     #     if mode == 'max':
#     #         score = -score
#     #
#     #     return {
#     #         'loss': score,
#     #         'space': space,
#     #         'status': hyperopt.STATUS_OK,
#     #         'model_params': model.params
#     #     }
#
#     # def _eval_model():
#     #     model.build()
#     #     model.compile()
#     #     model.fit(*train_pack.unpack(), verbose=verbose)
#     #     return model.evaluate(*test_pack.unpack(), verbose=verbose)
#     #
#     # if not model.params.hyper_space:
#     #     raise ValueError("Model hyper parameter space empty.")
#
#     trials = hyperopt.Trials()
#     hyperopt.fmin(
#         fn=_test_wrapper,
#         space=model.params.hyper_space,
#         algo=hyperopt.tpe.suggest,
#         max_evals=max_evals,
#         trials=trials
#     )
#     return [_clean_up_trial(trial) for trial in trials]


def _clean_up_trial(trial):
    return {
        'model_params': trial['result']['model_params'],
        'sampled_params': trial['result']['space'],
        'loss': trial['result']['loss']
    }
