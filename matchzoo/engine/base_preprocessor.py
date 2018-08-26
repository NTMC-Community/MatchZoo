""":class:`BasePreprocessor` define input and ouutput for processors."""

import abc
import typing
from pathlib import Path
import dill
import pandas as pd

from matchzoo import datapack


class BasePreprocessor(metaclass=abc.ABCMeta):
    """:class:`BasePreprocessor` to input handle data."""

    DATA_FILENAME = 'preprocessor.dill'

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
        data: datapack.DataPack,
        content: dict,
        context: dict,
    ) -> datapack.DataPack:
        """
        Return :class:`DataPack` instance as output.

        :param data: The :class:`DataPack` need to be preprocessed.
        :param content: Transformed output using preprocessor.
        :param context: Context to be passed to :class:`DataPack`.
        :param stage: Indicate the pre-processing stage, `train`
            or `test`.

        :return: Pre-processed input as well as context stored in
            a :class:`DataPack` object.
        """
        data.content = content
        data.context = context
        return data

    def save(self, dirpath: typing.Union[str, Path]):
        """
        Save the :class:`DSSMPreprocessor` object.

        A saved :class:`DSSMPreprocessor` is represented as a directory with
        the `context` object (fitted parameters on training data), it will
        be saved by `pickle`.

        :param dirpath: directory path of the saved :class:`DSSMPreprocessor`.
        """
        dirpath = Path(dirpath)
        data_file_path = dirpath.joinpath(self.DATA_FILENAME)

        if data_file_path.exists():
            raise FileExistsError
        elif not dirpath.exists():
            dirpath.mkdir()

        dill.dump(self, open(data_file_path, mode='wb'))

    def segmentation(self, inputs: list, stage: str) -> datapack.DataPack:
        """
        Convert user input into :class:`DataPack` consist of two tables.

        The `content` dict stores the id with it's corresponded input text.
        The `relation` table stores the relation between `text_left` and
            `text_right`.

        :param inputs: Raw user inputs, list of tuples.
        :param stage: `train` or `test`.

        :return: User input into a :class:`DataPack` with content and
            relation.
        """
        columns_all, columns_relation, columns_content = self._get_columns(
            stage=stage)

        # prepare data pack.
        inputs = pd.DataFrame(inputs, columns=columns_all)
        # get relation columns (idx left and idx right)
        relation = inputs[columns_relation].values

        # Segment input into 2 dataframes.
        content_left = inputs[['id_left', 'text_left']].drop_duplicates(
            ['id_left']
        )
        content_left.columns = columns_content

        content_right = inputs[['id_right', 'text_right']].drop_duplicates(
            ['id_right']
        )
        content_right.columns = columns_content

        content = pd.concat([content_left, content_right])

        content = content.set_index('id').to_dict(orient='index')

        return datapack.DataPack(relation=relation,
                                 content=content,
                                 columns=columns_relation)

    def _get_columns(self, stage: str) -> list:
        """Prepare columns for :class:`DataPack`."""
        columns_relation = ['id_left', 'id_right']
        columns_content = ['id', 'text']
        columns_all = ['id_left', 'id_right', 'text_left', 'text_right']

        if stage == 'train':
            columns_all.append('label')
            columns_relation.append('label')
        return columns_all, columns_relation, columns_content


def load_preprocessor(dirpath: typing.Union[str, Path]) -> datapack.DataPack:
    """
    Load the fitted `context`. The reverse function of :meth:`save`.

    :param dirpath: directory path of the saved model.
    :return: a :class:`DSSMPreprocessor` instance.
    """
    dirpath = Path(dirpath)

    data_file_path = dirpath.joinpath(BasePreprocessor.DATA_FILENAME)
    dp = dill.load(open(data_file_path, 'rb'))

    return dp
