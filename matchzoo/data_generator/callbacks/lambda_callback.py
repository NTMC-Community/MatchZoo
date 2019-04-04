from matchzoo.data_generator.callbacks.callback import Callback


class LambdaCallback(Callback):
    """
    LambdaCallback. Just a shorthand for creating a callback class.

    See :class:`matchzoo.data_generator.callbacks.Callback` for more details.

    Example:
        >>> from matchzoo.data_generator.callbacks import LambdaCallback
        >>> callback = LambdaCallback(on_batch_unpacked=print)
        >>> callback.on_batch_unpacked('x', 'y')
        x y

    """

    def __init__(self, on_batch_data_pack=None, on_batch_unpacked=None):
        """Init."""
        self._on_batch_unpacked = on_batch_unpacked
        self._on_batch_data_pack = on_batch_data_pack

    def on_batch_data_pack(self, data_pack):
        """`on_batch_data_pack`."""
        if self._on_batch_data_pack:
            self._on_batch_data_pack(data_pack)

    def on_batch_unpacked(self, x, y):
        """`on_batch_unpacked`."""
        if self._on_batch_unpacked:
            self._on_batch_unpacked(x, y)
