"""Director. Named with some flavor since I couldn't think of a better name."""

import logging

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


            # >>> results = director.action(verbose=0)
            # >>> sorted(results[0].keys())
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

    def action(self, verbose=2) -> list:
        """
        Start doing things.

        :param verbose: Verbosity. 0: None. 1: Some. 2: Full.
        :return: A list of trials.
        """
        all_trials = []
        num_models = len(self._params['models'])
        for i, model in enumerate(self._params['models']):
            show_stages = 1 if verbose >= 1 else 0
            show_details = 1 if verbose >= 2 else 0

            if not show_details:
                logging.getLogger('hyperopt').setLevel(logging.CRITICAL)

            preprocessor = model.get_default_preprocessor()
            preprocessor.fit(self._params['train_pack'], verbose=show_details)

            train_pack_processed = preprocessor.transform(
                self._params['train_pack'], verbose=show_details)
            test_pack_processed = preprocessor.transform(
                self._params['test_pack'], verbose=show_details)

            context = self._build_context(model, preprocessor)

            if show_stages:
                print(f"Start tunning model #{i + 1} (total: {num_models}).")
                print(f"Model class: {model.params['model_class']}")

            trials = tune(
                model=model,
                train_pack=train_pack_processed,
                test_pack=test_pack_processed,
                max_evals=self._params['evals_per_model'],
                context=context,
                verbose=show_details
            )

            if show_stages:
                print(f'Finish tuning model #{i + 1} (total: {num_models})')
                print()

            all_trials.extend(trials)

        return all_trials

    @classmethod
    def _build_context(cls, model, preprocessor):
        context = {}
        if 'input_shapes' in preprocessor.context:
            context['input_shapes'] = preprocessor.context['input_shapes']
        elif 'input_shapes' in model.params:
            context['input_shapes'] = model.params['input_shapes']
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        context['vocab_size'] = len(term_index) + 1
        return context
