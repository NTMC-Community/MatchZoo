import matchzoo as mz
from matchzoo.data_generator.data_generator import DataGenerator


class DataGeneratorBuilder(object):
    """
    Data Generator Bulider. In essense a wrapped partial function.

    Example:
        >>> import matchzoo as mz
        >>> builder = mz.DataGeneratorBuilder(mode='pair', batch_size=32)
        >>> data = mz.datasets.toy.load_data()
        >>> gen = builder.build(data)
        >>> type(gen)
        <class 'matchzoo.data_generator.data_generator.DataGenerator'>
        >>> gen.batch_size
        32
        >>> gen_64 = builder.build(data, batch_size=64)
        >>> gen_64.batch_size
        64

    """

    def __init__(self, **kwargs):
        """Init."""
        self._kwargs = kwargs

    def build(self, data_pack, **kwargs) -> DataGenerator:
        """
        Build a DataGenerator.

        :param data_pack: DataPack to build upon.
        :param kwargs: Additional keyword arguments to override the keyword
            arguments passed in `__init__`.
        """
        return mz.DataGenerator(data_pack, **{**self._kwargs, **kwargs})
