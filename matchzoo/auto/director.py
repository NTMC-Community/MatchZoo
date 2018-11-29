from matchzoo import engine
from .tune import tune


class Director(object):
    def __init__(self, params=None):
        self._params = params or self._get_default_params()

    @classmethod
    def _get_default_params(cls):
        params = engine.ParamTable()

        def _validate_models(models):
            return all(isinstance(m, engine.BaseModel) for m in models)

        params.add(engine.Param(
            'models', [], validator=_validate_models
        ))
        params.add(engine.Param('evals_per_model', 5))
        params.add(engine.Param('train_pack'))
        params.add(engine.Param('test_pack'))
        params.add(engine.Param('task'))
        return params

    @property
    def params(self):
        return self._params

    def action(self):
        all_trials = []
        for model in self._params['models']:
            preprocessor = model.get_default_preprocessor()
            preprocessor.fit(self._params['train_pack'])
            train_pack_processed = preprocessor.transform(
                self._params['train_pack'])
            test_pack_processed = preprocessor.transform(
                self._params['test_pack'])
            trials = tune(
                model=model,
                train_pack=train_pack_processed,
                test_pack=test_pack_processed,
                task=self._params['task'],
                max_evals=self._params['evals_per_model'],
                context=preprocessor.context
            )
            all_trials.append(trials)
        return all_trials
