"""Base generator."""

import abc


class BaseGenerator(abc.ABCMeta):
    """
    Matchzoo generator, accept datapack as input and produce samples for
    train and test.
    """

    def __init__(
            self,
            text_left_max_len: int,
            text_right_max_len: int,
            is_train: bool
            ):
        """Initialization."""
        self._text_left_max_len = text_left_max_len
        self._text_right_max_len = text_right_max_len
        self.is_train = is_train

    @abc.abstractmethod
    def get_data() -> tuple:
        """Get all the data instances."""

    @abc.abstractmethod
    def get_batch_generator() -> tuple:
        """Get a generator to produce samples dynamically."""
