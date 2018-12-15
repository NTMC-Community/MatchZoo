"""Basic Preprocessor."""

import logging

from tqdm import tqdm

from matchzoo import engine, processor_units
from matchzoo import DataPack
from matchzoo import chain_transform, build_vocab_unit, \
    build_unit_from_data_pack

logger = logging.getLogger(__name__)
tqdm.pandas()


class BasicPreprocessor(engine.BasePreprocessor):
    """
    Baisc preprocessor helper.

    :param fixed_length_left: Integer, maximize length of :attr:`left` in the
        data_pack.
    :param fixed_length_right: Integer, maximize length of :attr:`right` in the
        data_pack.

    Example:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_data('train')
        >>> test_data = mz.datasets.toy.load_data('test')
        >>> preprocessor = mz.preprocessors.BasicPreprocessor(
        ...     fixed_length_left=10,
        ...     fixed_length_right=20
        ... )
        >>> preprocessor = preprocessor.fit(train_data)
        >>> preprocessor.context['input_shapes']
        [(10,), (20,)]
        >>> preprocessor.context['vocab_size']
        284
        >>> processed_train_data = preprocessor.transform(train_data)
        >>> type(processed_train_data)
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        >>> test_data_transformed = preprocessor.transform(test_data)
        >>> type(test_data_transformed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>

    """

    def __init__(self, fixed_length_left: int = 30,
                 fixed_length_right: int = 30):
        """Initialization."""
        super().__init__()
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._left_fixedlength_unit = processor_units.FixedLengthUnit(
            self._fixed_length_left, pad_mode='post')
        self._right_fixedlength_unit = processor_units.FixedLengthUnit(
            self._fixed_length_right, pad_mode='post')

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
        self._context['vocab_size'] = len(vocab_unit.state['term_index'])

        self._context['input_shapes'] = [(self._fixed_length_left,),
                                         (self._fixed_length_right,)]

        return self

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
