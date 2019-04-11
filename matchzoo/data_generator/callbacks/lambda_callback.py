from matchzoo.data_generator.callbacks.callback import Callback


class LambdaCallback(Callback):
    """
    LambdaCallback. Just a shorthand for creating a callback class.

    See :class:`matchzoo.data_generator.callbacks.Callback` for more details.

    Example:

        >>> import matchzoo as mz
        >>> from matchzoo.data_generator.callbacks import LambdaCallback
        >>> data = mz.datasets.toy.load_data()
        >>> batch_func = lambda x: print(type(x))
        >>> unpack_func = lambda x, y: print(type(x), type(y))
        >>> callback = LambdaCallback(on_batch_data_pack=batch_func,
        ...                           on_batch_unpacked=unpack_func)
        >>> data_gen = mz.DataGenerator(
        ...     data, batch_size=len(data), callbacks=[callback])
        >>> _ = data_gen[0]
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        <class 'dict'> <class 'numpy.ndarray'>

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
