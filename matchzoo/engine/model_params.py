"""Model parametrs."""


class ModelParams(dict):
    """
    Model parametrs.

    All values of the keys are initialized to `None`. Inherited from `dict`
    with minimum changes.

    Example:

        >>> params = ModelParams()
        >>> list(params.items())
        [('name', None), ('model_class', None), ('input_shapes', None), \
('task', None), ('optimizer', None), ('loss', None), ('metrics', None)]

    """

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
