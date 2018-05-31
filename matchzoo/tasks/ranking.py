"""Ranking task."""

from matchzoo import engine


class Ranking(engine.BaseTask):
    """Ranking Task."""

    @classmethod
    def list_available_losses(cls) -> list:
        """Return a list of available losses."""
        return ['mse']

    @classmethod
    def list_available_metrics(cls) -> list:
        """Return a list of available metrics."""
        return ['mae']
