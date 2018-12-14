"""DRMMTKS Preprocessor."""

import logging

from tqdm import tqdm

from matchzoo import engine, processor_units
from matchzoo import DataPack
from matchzoo import chain_transform, build_vocab_unit, \
    build_unit_from_data_pack

logger = logging.getLogger(__name__)
tqdm.pandas()


class DRMMTKSPreprocessor(engine.BasePreprocessor):
    """
    DRMMTKS preprocessor helper.

    :param fixed_length: A pair of integers denote to the maximize length of
           :attr:`left` and :attr:`right` in the data_pack.

    Example:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_data('train')
        >>> test_data = mz.datasets.toy.load_data('test')
        >>> preprocessor = mz.preprocessors.DRMMTKSPreprocessor()
        >>> train_data_processed = preprocessor.fit_transform(train_data)
        >>> type(train_data_processed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        >>> test_data_transformed = preprocessor.transform(test_data)
        >>> type(test_data_transformed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>

    """

    def __init__(self, fixed_length: list = [10, 20]):
        """Initialization."""
        self._fixed_length = fixed_length
        self._left_fixedlength_unit = processor_units.FixedLengthUnit(
            self._fixed_length[0], pad_mode='post')
        self._right_fixedlength_unit = processor_units.FixedLengthUnit(
            self._fixed_length[1], pad_mode='post')
        super().__init__()

    def fit(self, data_pack: DataPack, verbose=1):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :param verbose: Verbosity.
        :return: class:`DRMMTKSPreprocessor` instance.
        """
        units = self._default_processor_units()
        data_pack = data_pack.apply_on_text(chain_transform(units))

        filter_unit = processor_units.FrequencyFilterUnit(low=2, mode='df')
        filter_unit = build_unit_from_data_pack(filter_unit, data_pack,
                                                flatten=False, mode='right',
                                                verbose=verbose)
        data_pack = data_pack.apply_on_text(filter_unit.transform,
                                            mode='right', verbose=verbose)
        self._context['filter_unit'] = filter_unit

        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)
        self._context['vocab_unit'] = vocab_unit
        self._context['embedding_input_dim'] = len(
            vocab_unit.state['term_index']) + 1

        self._context['input_shapes'] = [(self._fixed_length[0],),
                                         (self._fixed_length[1],)]

        return self

    def fit_transform(self, data_pack: DataPack, verbose=1) -> DataPack:
        """
        Fit the pre-processor and transform the :class:`DataPack` object.

        :param data_pack: :class:`DataPack` object to be processed.
        :param verbose: Verbosity.
        :return: the transformed :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        units = self._default_processor_units()
        data_pack.apply_on_text(chain_transform(units), inplace=True,
                                verbose=verbose)

        filter_unit = processor_units.FrequencyFilterUnit(low=2, mode='df')
        filter_unit = build_unit_from_data_pack(filter_unit, data_pack,
                                                flatten=False, mode='right',
                                                verbose=verbose)
        data_pack.apply_on_text(filter_unit.transform, mode='right',
                                inplace=True)
        self._context['filter_unit'] = filter_unit

        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)
        data_pack.apply_on_text(vocab_unit.transform, inplace=True,
                                mode='both', verbose=verbose)
        self._context['vocab_unit'] = vocab_unit

        data_pack.append_text_length(inplace=True)
        data_pack.apply_on_text(self._left_fixedlength_unit.transform,
                                mode='left', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._right_fixedlength_unit.transform,
                                mode='right', inplace=True, verbose=verbose)

        self._context['embedding_input_dim'] = len(
            vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = [(self._fixed_length[0],),
                                         (self._fixed_length[1],)]
        return data_pack

    @engine.validate_context
    def transform(self, data_pack: DataPack, verbose=1) -> DataPack:
        """
        Apply transformation on data, create fixed length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        units = self._default_processor_units()
        data_pack.apply_on_text(chain_transform(units), inplace=True,
                                verbose=verbose)

        data_pack.apply_on_text(self._context['filter_unit'].transform,
                                mode='right', inplace=True, verbose=verbose)

        data_pack.apply_on_text(self._context['vocab_unit'].transform,
                                mode='both', inplace=True, verbose=verbose)

        data_pack.append_text_length(inplace=True)
        data_pack.apply_on_text(self._left_fixedlength_unit.transform,
                                mode='left', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._right_fixedlength_unit.transform,
                                mode='right', inplace=True, verbose=verbose)
        return data_pack

    @classmethod
    def _default_processor_units(cls) -> list:
        """Prepare needed process units."""
        return [
            processor_units.TokenizeUnit(),
            processor_units.LowercaseUnit(),
            processor_units.PuncRemovalUnit(),
        ]
