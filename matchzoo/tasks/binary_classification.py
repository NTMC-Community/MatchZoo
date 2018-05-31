"""Binary classification task."""

from matchzoo import engine


class BinaryClassification(engine.BaseTask):
    """Classification task."""

    @classmethod
    def list_available_losses(cls) -> list:
        """Return a list of available losses."""
        return ['binary_crossentropy']

    @classmethod
    def list_available_metrics(cls) -> list:
        """Return a list of available metrics."""
        return ['acc']
