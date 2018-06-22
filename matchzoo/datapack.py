"""Matchzoo DataPack, paiir-wise tuple (feature) and context as input."""

import typing
import pandas as pd


class DataPack(pd.DataFrame):
    """
    Matchzoo DataPack data structure, store dataframe and context.

    Example:
        >>> # features, context generate by processors.
        >>> features = [([1,3], [2,3]), ([3,0], [1,6])]
        >>> context = {'vocab_size': 2000}
        >>> dp = DataPack(data=features,
        ...               context=context)
        >>> dp.context
        {'vocab_size': 2000}
        >>> # sample without replacement for generation.
        >>> type(dp.sample(1))
        <class 'matchzoo.datapack.DataPack'>
        >>> dp.size
        2
        >>> features, context = dp.unpack()
    """

    _metadata = ['context']

    def __init__(self,
                 data: list,
                 context: dict={},
                 index: list= None,
                 columns: list=['text_left', 'text_right'],
                 dtype: object=None,
                 copy: bool=True):
        """Initialize."""
        super(DataPack, self).__init__(data=data,
                                       index=index,
                                       columns=columns,
                                       dtype=dtype,
                                       copy=copy)
        self.context = context

    @property
    def _constructor(self) -> callable:
        """Subclass pd.DataFrame."""
        return DataPack._internal_ctor

    @classmethod
    def _internal_ctor(cls, *args, **kwargs):
        """Create subclass inputs to store context."""
        kwargs['context'] = None
        return cls(*args, **kwargs)

    @property
    def size(self) -> int:
        """Get size of the data pack."""
        return self.shape[0]

    def unpack(self) -> typing.Union[pd.DataFrame, dict]:
        """Unpack DataPack."""
        return self, self.context
