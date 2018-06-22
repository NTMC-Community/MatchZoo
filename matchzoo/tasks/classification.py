"""Classification task."""

import keras

from matchzoo import engine


class Classification(engine.BaseTask):
    """Classification task."""

    def __init__(self, num_classes: int = 2):
        """Classification task."""
        super().__init__()
        if not isinstance(num_classes, int):
            raise TypeError("Number of classes must be an integer.")
        if num_classes < 2:
            raise ValueError("Number of classes can't be smaller than 2")
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        """:return: number of classes to classify."""
        return self._num_classes

    @classmethod
    def list_available_losses(cls) -> list:
        """:return: a list of available losses."""
        return ['categorical_crossentropy']

    @classmethod
    def list_available_metrics(cls) -> list:
        """:return: a list of available metrics."""
        return ['acc']

    def make_output_layer(self) -> keras.layers.Dense:
        """:return: a correctly shaped keras dense layer for model output."""
        return keras.layers.Dense(self._num_classes, activation='softmax')

    @property
    def output_shape(self) -> tuple:
        """:return: output shape of a single sample of the task."""
        return self._num_classes,
