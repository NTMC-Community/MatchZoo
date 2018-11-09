"""DSSM Preprocessor."""

import logging

from tqdm import tqdm

from matchzoo import DataPack
from matchzoo import engine
from matchzoo import preprocessors

logger = logging.getLogger(__name__)
tqdm.pandas()


class DSSMPreprocessor(engine.BasePreprocessor):
    """
    DSSM preprocessor helper.

    TODO: NEED REFACTORING.

    Example:
        >>> train_inputs = [
        ...     ("id0", "id1", "beijing", "Beijing is capital of China", 1),
        ...     ("id0", "id2", "beijing", "China is in east Asia", 0),
        ...     ("id0", "id3", "beijing", "Summer in Beijing is hot.", 1)
        ... ]
        >>> dssm_preprocessor = DSSMPreprocessor()
        >>> rv_train = dssm_preprocessor.fit_transform(
        ...     train_inputs,
        ...     stage='train')
        >>> type(rv_train)
        <class 'matchzoo.data_pack.DataPack'>
        >>> test_inputs = [("id0",
        ...                 "id4",
        ...                 "beijing",
        ...                 "I visted beijing yesterday.")]
        >>> rv_test = dssm_preprocessor.fit_transform(
        ...     test_inputs,
        ...     stage='predict')
        >>> type(rv_test)
        <class 'matchzoo.data_pack.DataPack'>

    """

    def fit(self, data_pack: DataPack):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :return: class:`DSSMPreprocessor` instance.
        """

        vocab = []
        units = self._preprocess_units()

        def transform_and_extend_vocab(text):
            for unit in units:
                text = unit.transform(text)
            vocab.extend(text)

        data_pack.apply_on_text(transform_and_extend_vocab)

        vocab_unit = preprocessors.VocabularyUnit()
        vocab_unit.fit(tqdm(vocab, desc='Fitting vocabulary unit.'))

        self._context.update(vocab_unit.state)
        dim_triletter = len(vocab_unit.state['term_index']) + 1
        self._context['input_shapes'] = [(dim_triletter,), (dim_triletter,)]

        return self

    @engine.validate_context
    def transform(self, data_pack: DataPack) -> DataPack:
        """
        Apply transformation on data, create `tri-letter` representation.

        :param data_pack: Inputs to be preprocessed.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        term_index = self._context['term_index']
        hashing_unit = preprocessors.WordHashingUnit(term_index)
        units = self._preprocess_units() + [hashing_unit]
        for unit in units:
            data_pack.apply_on_text(unit.transform, inplace=True)
        return data_pack

    @classmethod
    def _preprocess_units(cls) -> list:
        """Prepare needed process units."""
        return [
            preprocessors.TokenizeUnit(),
            preprocessors.LowercaseUnit(),
            preprocessors.PuncRemovalUnit(),
            preprocessors.StopRemovalUnit(),
            preprocessors.NgramLetterUnit(),
        ]