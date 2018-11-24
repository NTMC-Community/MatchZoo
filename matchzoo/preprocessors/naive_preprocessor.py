"""DSSM Preprocessor."""

import logging

from tqdm import tqdm

from matchzoo import engine, processor_units
from matchzoo import DataPack
from matchzoo import chain_transform, build_vocab

logger = logging.getLogger(__name__)
tqdm.pandas()


class NaivePreprocessor(engine.BasePreprocessor):
    """
    Naive preprocessor.

    Example:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_train_classify_data()
        >>> test_data = mz.datasets.toy.load_test_classify_data()
        >>> preprocessor = mz.preprocessors.NaivePreprocessor()
        >>> train_data_processed = preprocessor.fit_transform(train_data)
        >>> type(train_data_processed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        >>> test_data_transformed = preprocessor.transform(test_data)
        >>> type(test_data_transformed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>

    """

    def fit(self, data_pack: DataPack):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :return: class:`DSSMPreprocessor` instance.
        """
        units = self._default_processor_units()
        data_pack = data_pack.apply_on_text(chain_transform(units))
        vocab_unit = build_vocab(data_pack)
        self._context.update(vocab_unit.state)
        return self

    @engine.validate_context
    def transform(self, data_pack: DataPack) -> DataPack:
        """
        Apply transformation on data, create `tri-letter` representation.

        :param data_pack: Inputs to be preprocessed.

        :return: Transformed data as :class:`DataPack` object.
        """
        units = self._default_processor_units()
        return data_pack.apply_on_text(chain_transform(units))
