"""DRMMTKS Preprocessor."""

import logging

from tqdm import tqdm

from matchzoo import engine, preprocessors, processor_units
from matchzoo import DataPack
from matchzoo import chain_transform, build_vocab

logger = logging.getLogger(__name__)
tqdm.pandas()


class DRMMTKSPreprocessor(engine.BasePreprocessor):
    """
    DRMMTKS preprocessor helper.

    TODO: NEED REFACTORING.

    Example:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_train_classify_data()
        >>> test_data = mz.datasets.toy.load_test_classify_data()
        >>> drmmtks_preprocessor = mz.preprocessors.DRMMTKSPreprocessor()
        >>> train_data_processed = drmmtks_preprocessor.fit_transform(train_data)
        >>> type(train_data_processed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        >>> test_data_transformed = drmmtks_preprocessor.transform(test_data)
        >>> type(test_data_transformed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>

    """

    def __init__(self, fixed_length: list=[10, 20]):
        """Initialization."""
        self._fixed_length = fixed_length
        self._left_fixedlength_unit = processor_units.FixedLengthUnit(
            self._fixed_length[0])
        self._right_fixedlength_unit = processor_units.FixedLengthUnit(
            self._fixed_length[1])
        super().__init__()

    def fit(self, data_pack: DataPack):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :return: class:`DRMMTKSPreprocessor` instance.
        """
        units = self._default_processor_units()
        data_pack = data_pack.apply_on_text(chain_transform(units))
        self.vocab_unit = build_vocab(data_pack)

        #self._context.update(vocab_unit.state)
        #self._context['vocab_unit'] = vocab_unit
        self._context['term_index'] = self.vocab_unit.state['term_index']
        self._context['vocab_size'] = len(self._context['term_index']) + 1
        self._context['input_shapes'] = [(self._fixed_length[0],),
                                         (self._fixed_length[1],)]

        return self

    @engine.validate_context
    def transform(self, data_pack: DataPack) -> DataPack:
        """
        Apply transformation on data, create `tri-letter` representation.

        :param data_pack: Inputs to be preprocessed.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        units = self._default_processor_units() + [self.vocab_unit]
        data_pack.apply_on_text(chain_transform(units), inplace=True)
        data_pack.apply_on_left(
            chain_transform([self._left_fixedlength_unit]),
            name='text_left',
            inplace=True)
        data_pack.apply_on_right(
            chain_transform([self._right_fixedlength_unit]),
            name='text_right',
            inplace=True)
        return data_pack

    @classmethod
    def _default_processor_units(cls) -> list:
        """Prepare needed process units."""
        return [
            processor_units.TokenizeUnit(),
            processor_units.LowercaseUnit(),
            processor_units.PuncRemovalUnit(),
            processor_units.StopRemovalUnit(),
        ]
