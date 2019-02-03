import matchzoo
from matchzoo.engine.base_model import BaseModel


class Callback(object):
    """
    Tuner callback base class.

    To build your own callbacks, inherit `mz.auto.tuner.callbacks.Callback`
    and overrides corresponding methods.

    A run proceeds in the following way:

    - run start (callback)
    - build model
    - build end (callback)
    - fit and evaluate model
    - collect result
    - run end (callback)

    This process is repeated for `num_runs` times in a tuner.

    """

    def on_run_start(self, tuner: 'matchzoo.auto.Tuner', sample: dict):
        """
        Callback on run start stage.

        :param tuner: Tuner.
        :param sample: Sampled hyper space. Changes to this dictionary affects
            the model building process of the tuner.
        """

    def on_build_end(self, tuner: 'matchzoo.auto.Tuner', model: BaseModel):
        """
        Callback on build end stage.

        :param tuner: Tuner.
        :param model: A built model ready for fitting and evluating. Changes
            to this model affect the fitting and evaluating process.
        """

    def on_run_end(self, tuner: 'matchzoo.auto.Tuner', model: BaseModel,
                   result: dict):
        """
        Callback on run end stage.

        :param tuner: Tuner.
        :param model: A built model done fitting and evaluating. Changes to
            the model will no longer affect the result.
        :param result: Result of the run. Changes to this dictionary will be
            visible in the return value of the `tune` method.
        """
