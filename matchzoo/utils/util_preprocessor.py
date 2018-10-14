"""Utils for preprocessors."""


def validate_context(func):
    """Validate context in the preprocessor."""
    def transform_wrapper(self, *args, **kwargs):
        if not self.context:
            raise ValueError(
                'Please fit parameters before transformation.')
        return func(self, *args, **kwargs)
    return transform_wrapper
