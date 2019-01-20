import numpy as np

import matchzoo as mz


class Callback(object):
    """
    DataGenerator callback base class.

    To build your own callbacks, inherit `mz.data_generator.callbacks.Callback`
    and overrides corresponding methods.

    A batch is processed in the following way:

    - slice data pack based on batch index
    - handle `on_batch_data_pack` callbacks
    - unpack data pack into x, y
    - handle `on_batch_x_y` callbacks
    - return x, y

    """

    def on_batch_data_pack(self, data_pack: mz.DataPack):
        """
        `on_batch_data_pack`.

        :param data_pack: a sliced DataPack before unpacking.
        """

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """
        `on_batch_unpacked`.

        :param x: unpacked x.
        :param y: unpacked y.
        """
