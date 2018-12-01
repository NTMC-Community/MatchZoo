"""Director. Named with some flavor since I couldn't think of a better name."""

import typing

from matchzoo import engine
from matchzoo.auto.tune import tune


class Director(object):
    """Director."""

    def __init__(self, params=None):
        """
        Director automizes preprocessing, parameter searching and training.

        Example:
            >>> import matchzoo as mz
            >>> director = mz.Director()
            >>> sorted(director.params.keys())
            ['evals_per_model', 'models', 'task', 'test_pack', 'train_pack']
            >>> director.params['evals_per_model'] = 3
            >>> director.params['models'] = [
            ...     mz.models.DenseBaselineModel(),
            ...     mz.models.DSSMModel()
            ... ]
            >>> data_pack = mz.datasets.toy.load_train_rank_data()
            >>> split = int(len(data_pack) * 0.9)
            >>> train_pack = data_pack[:split]
            >>> test_pack = data_pack[split:]
            >>> director.params['train_pack'] = train_pack
            >>> director.params['test_pack'] = test_pack
            >>> director.params['task'] = mz.tasks.Ranking()
            >>> results = director.action()
            >>> len(results) == len(director.params['models'])
            True
            >>> len(results[0]) == director.params['evals_per_model']
            True
            >>> sorted(results[0][0].keys())
            ['loss', 'model_params', 'sampled_params']

        """
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
    def params(self) -> engine.ParamTable:
        """:return: Parameters."""
        return self._params

    def action(self) -> typing.List[typing.List[typing.Dict[str, typing.Any]]]:
        """:return: a list of trials."""
        all_trials = []
        for model in self._params['models']:
            preprocessor = model.get_default_preprocessor()
            preprocessor.fit(self._params['train_pack'], verbose=0)

            train_pack_processed = preprocessor.transform(
                self._params['train_pack'], verbose=0)
            test_pack_processed = preprocessor.transform(
                self._params['test_pack'], verbose=0)

            trials = tune(
                model=model,
                train_pack=train_pack_processed,
                test_pack=test_pack_processed,
                task=self._params['task'],
                max_evals=self._params['evals_per_model'],
                context=preprocessor.context,
                verbose=0
            )

            all_trials.append(trials)

        return all_trials
