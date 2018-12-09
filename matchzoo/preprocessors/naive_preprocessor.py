"""DSSM Preprocessor."""

import logging

from tqdm import tqdm

from matchzoo import engine, processor_units
from matchzoo import DataPack
from matchzoo import chain_transform, build_vocab_unit, \
    build_unit_from_data_pack

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

    def fit(self, data_pack: DataPack, verbose=1):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :param verbose: Verbosity.
        :return: class:`DSSMPreprocessor` instance.
        """
        units = self._default_processor_units()
        data_pack = data_pack.apply_on_text(chain_transform(units),
                                            verbose=verbose)
        filter_unit = processor_units.FrequencyFilterUnit(low=2, mode='df')
        filter_unit = build_unit_from_data_pack(filter_unit, data_pack,
                                                flatten=False, verbose=verbose)
        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)
        self._context['filter_unit'] = filter_unit
        self._context['vocab_unit'] = vocab_unit
        return self

    @engine.validate_context
    def transform(self, data_pack: DataPack, verbose=1) -> DataPack:
        """
        Apply transformation on data, create `tri-letter` representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        units = self._default_processor_units()
        units.append(self._context['filter_unit'])
        units.append(self._context['vocab_unit'])
        units.append(processor_units.FixedLengthUnit(text_length=30))
        return data_pack.apply_on_text(chain_transform(units), verbose=verbose)
