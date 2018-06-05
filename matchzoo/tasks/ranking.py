"""Ranking task."""

import keras

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

    def make_output_layer(self):
        """:return: a correctly shaped keras dense layer for model output."""
        return keras.layers.Dense(1, activation='sigmoid')
