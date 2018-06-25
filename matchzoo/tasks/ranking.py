"""Ranking task."""

from matchzoo import engine


class Ranking(engine.BaseTask):
    """Ranking Task."""

    @classmethod
    def list_available_losses(cls) -> list:
        """:return: a list of available losses."""
        return ['mse']

    @classmethod
    def list_available_metrics(cls) -> list:
        """:return: a list of available metrics."""
        return ['mae']

    @property
    def output_shape(self) -> tuple:
        """:return: output shape of a single sample of the task."""
        return 1,
