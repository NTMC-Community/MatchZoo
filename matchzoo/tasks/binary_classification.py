from matchzoo import engine


class BinaryClassification(engine.BaseTask):
    """Classification Task."""

    @classmethod
    def list_available_losses(cls) -> list:
        return ['binary_crossentropy']

    @classmethod
    def list_available_metrics(cls) -> list:
        return ['acc']
