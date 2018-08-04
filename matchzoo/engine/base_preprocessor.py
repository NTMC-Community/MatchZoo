""":class:`BasePreprocessor` define input and ouutput for processors."""

import abc
from matchzoo import datapack


class BasePreprocessor(metaclass=abc.ABCMeta):
    """:class:`BasePreprocessor` to input handle data."""

    @abc.abstractmethod
    def fit(self, inputs: list) -> 'BasePreprocessor':
        """
        Fit parameters on input data.

        This method is an abstract base method, need to be
        implemented in the child class.

        This method is expected to return itself as a callable
        object.

        :param inputs: List of text-left, text-right, label triples.
        """

    @abc.abstractmethod
    def transform(self, inputs: list, stage: str) -> datapack.DataPack:
        """
        Transform input data to expected manner.

        This method is an abstract base method, need to be
        implemented in the child class.

        :param inputs: List of text-left, text-right, label triples,
            or list of text-left, text-right tuples (test stage).
        :param stage: String indicate the pre-processing stage, `train` or
            `test` expected.
        """

    def fit_transform(self, inputs: list, stage: str) -> datapack.DataPack:
        """
        Call fit-transform.

        :param inputs: List of text-left, text-right, label triples.
        :param stage: String indicate the pre-processing stage, `train` or
            `test` expected.
        """
        if stage == 'train':
            return self.fit(inputs).transform(inputs, stage)
        else:
            return self.transform(inputs, stage)

    def _make_output(
        self,
        output: list,
        context: dict,
        stage: str
    ) -> datapack.DataPack:
        """
        Create :class:`DataPack` instance as output.

        :param output: Transformed output using preprocessor.
        :param context: Context to be passed to :class:`DataPack`.
        :param stage: Indicate the pre-processing stage, `train`
            or `test`.
        """
        columns = ['id_left', 'id_right', 'text_left', 'text_right']
        if stage == 'train':
            columns.append('label')
        return datapack.DataPack(output, context, columns)
