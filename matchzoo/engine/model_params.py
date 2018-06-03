"""Model parametrs."""


class ModelParams(dict):
    """Model parametrs."""

    def __init__(self):
        """Model parametrs."""

        super().__init__(
                name=None,
                model_class=None,
                input_shapes=None,
                task=None,
                optimizer=None,
                loss=None,
                metrics=None,
        )
