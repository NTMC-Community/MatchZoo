from matchzoo.data_generator.callbacks.callback import Callback


class LambdaCallback(Callback):
    def __init__(self, on_batch_data_pack=None, on_batch_unpacked=None):
        self._on_batch_unpacked = on_batch_unpacked
        self._on_batch_data_pack = on_batch_data_pack

    def on_batch_data_pack(self, data_pack):
        self._on_batch_data_pack(data_pack)

    def on_batch_unpacked(self, x, y):
        self._on_batch_unpacked(x, y)
