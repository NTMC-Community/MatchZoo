class BaseModelParams(dict):
    def __init__(self):
        super().__init__(
                name=None,
                input_shapes=None,
                task=None,
                optimizer=None,
                loss=None,
                metrics=None,
        )
