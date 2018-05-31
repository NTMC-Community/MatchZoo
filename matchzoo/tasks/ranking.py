from matchzoo import engine


class Ranking(engine.BaseTask):
    """Ranking Task."""

    @classmethod
    def list_available_losses(cls) -> list:
        return ['mse']

    @classmethod
    def list_available_metrics(cls) -> list:
        return ['mae']
