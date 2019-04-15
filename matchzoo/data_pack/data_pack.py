"""Matchzoo DataPack, pair-wise tuple (feature) and context as input."""

import typing
import inspect
from pathlib import Path
import functools

import dill
from tqdm import tqdm
import numpy as np
import pandas as pd

import matchzoo

tqdm.pandas()


def _convert_to_list_index(
    index: typing.Union[int, slice, np.array],
    length: int
):
    if isinstance(index, int):
        index = [index]
    elif isinstance(index, slice):
        index = list(range(*index.indices(length)))
    return index


class DataPack(object):
    """
    Matchzoo :class:`DataPack` data structure, store dataframe and context.

    `DataPack` is a MatchZoo native data structure that most MatchZoo data
    handling processes build upon. A `DataPack` consists of three parts:
    `left`, `right` and `relation`, each one of is a `pandas.DataFrame`.

    :param relation: Store the relation between left document
        and right document use ids.
    :param left: Store the content or features for id_left.
    :param right: Store the content or features for
        id_right.

    Example:
        >>> left = [
        ...     ['qid1', 'query 1'],
        ...     ['qid2', 'query 2']
        ... ]
        >>> right = [
        ...     ['did1', 'document 1'],
        ...     ['did2', 'document 2']
        ... ]
        >>> relation = [['qid1', 'did1', 1], ['qid2', 'did2', 1]]
        >>> relation_df = pd.DataFrame(relation)
        >>> left = pd.DataFrame(left)
        >>> right = pd.DataFrame(right)
        >>> dp = DataPack(
        ...     relation=relation_df,
        ...     left=left,
        ...     right=right,
        ... )
        >>> len(dp)
        2
    """

    DATA_FILENAME = 'data.dill'

    def __init__(
        self,
        relation: pd.DataFrame,
        left: pd.DataFrame,
        right: pd.DataFrame
    ):
        """:class:`DataPack` initializer."""
        self._relation = relation
        self._left = left
        self._right = right

    @property
    def has_label(self) -> bool:
        """:return: `True` if `label` column exists, `False` other wise."""
        return 'label' in self._relation.columns

    def __len__(self) -> int:
        """Get numer of rows in the class:`DataPack` object."""
        return self._relation.shape[0]

    @property
    def frame(self) -> 'DataPack.FrameView':
        """
        View the data pack as a :class:`pandas.DataFrame`.

        Returned data frame is created by merging the left data frame,
        the right dataframe and the relation data frame. Use `[]` to access
        an item or a slice of items.

        :return: A :class:`matchzoo.DataPack.FrameView` instance.

        Example:
            >>> import matchzoo as mz
            >>> data_pack = mz.datasets.toy.load_data()
            >>> type(data_pack.frame)
            <class 'matchzoo.data_pack.data_pack.DataPack.FrameView'>
            >>> frame_slice = data_pack.frame[0:5]
            >>> type(frame_slice)
            <class 'pandas.core.frame.DataFrame'>
            >>> list(frame_slice.columns)
            ['id_left', 'text_left', 'id_right', 'text_right', 'label']
            >>> full_frame = data_pack.frame()
            >>> len(full_frame) == len(data_pack)
            True

        """
        return DataPack.FrameView(self)

    def unpack(self) -> typing.Tuple[typing.Dict[str, np.array],
                                     typing.Optional[np.array]]:
        """
        Unpack the data for training.

        The return value can be directly feed to `model.fit` or
        `model.fit_generator`.

        :return: A tuple of (X, y). `y` is `None` if `self` has no label.

        Example:
            >>> import matchzoo as mz
            >>> data_pack = mz.datasets.toy.load_data()
            >>> X, y = data_pack.unpack()
            >>> type(X)
            <class 'dict'>
            >>> sorted(X.keys())
            ['id_left', 'id_right', 'text_left', 'text_right']
            >>> type(y)
            <class 'numpy.ndarray'>
            >>> X, y = data_pack.drop_label().unpack()
            >>> type(y)
            <class 'NoneType'>

        """
        frame = self.frame()

        columns = list(frame.columns)
        if self.has_label:
            columns.remove('label')
            y = np.vstack(np.asarray(frame['label']))
        else:
            y = None

        x = frame[columns].to_dict(orient='list')
        for key, val in x.items():
            x[key] = np.array(val)

        return x, y

    def __getitem__(self, index: typing.Union[int, slice, np.array]
                    ) -> 'DataPack':
        """
        Get specific item(s) as a new :class:`DataPack`.

        The returned :class:`DataPack` will be a copy of the subset of the
        original :class:`DataPack`.

        :param index: Index of the item(s) to get.
        :return: An instance of :class:`DataPack`.
        """
        index = _convert_to_list_index(index, len(self))
        relation = self._relation.loc[index].reset_index(drop=True)
        left = self._left.loc[relation['id_left'].unique()]
        right = self._right.loc[relation['id_right'].unique()]
        return DataPack(left=left.copy(),
                        right=right.copy(),
                        relation=relation.copy())

    @property
    def relation(self):
        """`relation` getter."""
        return self._relation

    @relation.setter
    def relation(self, value):
        """`relation` setter."""
        self._relation = value

    @property
    def left(self) -> pd.DataFrame:
        """Get :meth:`left` of :class:`DataPack`."""
        return self._left

    @property
    def right(self) -> pd.DataFrame:
        """Get :meth:`right` of :class:`DataPack`."""
        return self._right

    def copy(self) -> 'DataPack':
        """:return: A deep copy."""
        return DataPack(left=self._left.copy(),
                        right=self._right.copy(),
                        relation=self._relation.copy())

    def save(self, dirpath: typing.Union[str, Path]):
        """
        Save the :class:`DataPack` object.

        A saved :class:`DataPack` is represented as a directory with a
        :class:`DataPack` object (transformed user input as features and
        context), it will be saved by `pickle`.

        :param dirpath: directory path of the saved :class:`DataPack`.
        """
        dirpath = Path(dirpath)
        data_file_path = dirpath.joinpath(self.DATA_FILENAME)

        if data_file_path.exists():
            raise FileExistsError(
                f'{data_file_path} already exist, fail to save')
        elif not dirpath.exists():
            dirpath.mkdir()

        dill.dump(self, open(data_file_path, mode='wb'))

    def _optional_inplace(func):
        """
        Decorator that adds `inplace` key word argument to a method.

        Decorate any method that modifies inplace to make that inplace change
        optional.
        """
        doc = ":param inplace: `True` to modify inplace, `False` to return " \
              "a modified copy. (default: `False`)"

        def _clean(s):
            return s.replace(' ', '').replace('\n', '')

        if _clean(doc) not in _clean(inspect.getdoc(func)):
            raise NotImplementedError(
                f"`inplace` parameter of {func} not documented.\n"
                f"Please add the following line to its documentation:\n{doc}")

        @functools.wraps(func)
        def wrapper(
            self, *args, inplace: bool = False, **kwargs
        ) -> typing.Optional['DataPack']:

            if inplace:
                target = self
            else:
                target = self.copy()

            func(target, *args, **kwargs)

            if not inplace:
                return target

        return wrapper

    @_optional_inplace
    def shuffle(self):
        """
        Shuffle the data pack by shuffling the relation column.

        :param inplace: `True` to modify inplace, `False` to return a modified
            copy. (default: `False`)

        Example:
            >>> import matchzoo as mz
            >>> import numpy.random
            >>> numpy.random.seed(0)
            >>> data_pack = mz.datasets.toy.load_data()
            >>> orig_ids = data_pack.relation['id_left']
            >>> shuffled = data_pack.shuffle()
            >>> (shuffled.relation['id_left'] != orig_ids).any()
            True

        """
        self._relation = self._relation.sample(frac=1)
        self._relation.reset_index(drop=True, inplace=True)

    @_optional_inplace
    def drop_label(self):
        """
        Remove `label` column from the data pack.

        :param inplace: `True` to modify inplace, `False` to return a modified
            copy. (default: `False`)

        Example:
            >>> import matchzoo as mz
            >>> data_pack = mz.datasets.toy.load_data()
            >>> data_pack.has_label
            True
            >>> data_pack.drop_label(inplace=True)
            >>> data_pack.has_label
            False
        """
        self._relation = self._relation.drop(columns='label')

    @_optional_inplace
    def append_text_length(self, verbose=1):
        """
        Append `length_left` and `length_right` columns.

        :param inplace: `True` to modify inplace, `False` to return a modified
            copy. (default: `False`)
        :param verbose: Verbosity.

        Example:
            >>> import matchzoo as mz
            >>> data_pack = mz.datasets.toy.load_data()
            >>> 'length_left' in data_pack.frame[0].columns
            False
            >>> new_data_pack = data_pack.append_text_length(verbose=0)
            >>> 'length_left' in new_data_pack.frame[0].columns
            True
            >>> 'length_left' in data_pack.frame[0].columns
            False
            >>> data_pack.append_text_length(inplace=True, verbose=0)
            >>> 'length_left' in data_pack.frame[0].columns
            True

        """
        self.apply_on_text(len, rename=('length_left', 'length_right'),
                           inplace=True, verbose=verbose)

    @_optional_inplace
    def apply_on_text(
        self, func: typing.Callable,
        mode: str = 'both',
        rename: typing.Optional[str] = None,
        verbose: int = 1
    ):
        """
        Apply `func` to text columns based on `mode`.

        :param func: The function to apply.
        :param mode: One of "both", "left" and "right".
        :param rename: If set, use new names for results instead of replacing
            the original columns. To set `rename` in "both" mode, use a tuple
            of `str`, e.g. ("text_left_new_name", "text_right_new_name").
        :param inplace: `True` to modify inplace, `False` to return a modified
            copy. (default: `False`)
        :param verbose: Verbosity.

        Examples::
            >>> import matchzoo as mz
            >>> data_pack = mz.datasets.toy.load_data()
            >>> frame = data_pack.frame

        To apply `len` on the left text and add the result as 'length_left':
            >>> data_pack.apply_on_text(len, mode='left',
            ...                         rename='length_left',
            ...                         inplace=True,
            ...                         verbose=0)
            >>> list(frame[0].columns) # noqa: E501
            ['id_left', 'text_left', 'length_left', 'id_right', 'text_right', 'label']

        To do the same to the right text:
            >>> data_pack.apply_on_text(len, mode='right',
            ...                         rename='length_right',
            ...                         inplace=True,
            ...                         verbose=0)
            >>> list(frame[0].columns) # noqa: E501
            ['id_left', 'text_left', 'length_left', 'id_right', 'text_right', 'length_right', 'label']

        To do the same to the both texts at the same time:
            >>> data_pack.apply_on_text(len, mode='both',
            ...                         rename=('extra_left', 'extra_right'),
            ...                         inplace=True,
            ...                         verbose=0)
            >>> list(frame[0].columns) # noqa: E501
            ['id_left', 'text_left', 'length_left', 'extra_left', 'id_right', 'text_right', 'length_right', 'extra_right', 'label']

        To suppress outputs:
            >>> data_pack.apply_on_text(len, mode='both', verbose=0,
            ...                         inplace=True)

        """
        if mode == 'both':
            self._apply_on_text_both(func, rename, verbose=verbose)
        elif mode == 'left':
            self._apply_on_text_left(func, rename, verbose=verbose)
        elif mode == 'right':
            self._apply_on_text_right(func, rename, verbose=verbose)
        else:
            raise ValueError(f"{mode} is not a valid mode type."
                             f"Must be one of `left` `right` `both`.")

    def _apply_on_text_right(self, func, rename, verbose=1):
        name = rename or 'text_right'
        if verbose:
            tqdm.pandas(desc="Processing " + name + " with " + func.__name__)
            self._right[name] = self._right['text_right'].progress_apply(func)
        else:
            self._right[name] = self._right['text_right'].apply(func)

    def _apply_on_text_left(self, func, rename, verbose=1):
        name = rename or 'text_left'
        if verbose:
            tqdm.pandas(desc="Processing " + name + " with " + func.__name__)
            self._left[name] = self._left['text_left'].progress_apply(func)
        else:
            self._left[name] = self._left['text_left'].apply(func)

    def _apply_on_text_both(self, func, rename, verbose=1):
        left_name, right_name = rename or ('text_left', 'text_right')
        self._apply_on_text_left(func, rename=left_name, verbose=verbose)
        self._apply_on_text_right(func, rename=right_name, verbose=verbose)

    @_optional_inplace
    def one_hot_encode_label(self, num_classes=2):
        """
        One-hot encode `label` column of `relation`.

        :param num_classes: Number of classes.
        :param inplace: `True` to modify inplace, `False` to return a modified
            copy. (default: `False`)
        :return:
        """
        self._relation['label'] = self._relation['label'].apply(
            lambda idx: matchzoo.one_hot(idx, num_classes))

    class FrameView(object):
        """FrameView."""

        def __init__(self, data_pack: 'DataPack'):
            """
            View a data pack as a frame.

            A slice of the view is genereated by merging three parts of the
            data pack being viewed into a big table.

            :param data_pack: :class:`DataPack` to view.

            Examples::
                >>> import matchzoo as mz
                >>> data_pack = mz.datasets.toy.load_data()
                >>> frame = data_pack.frame

            Use `()` to get a full copy of the frame:
                >>> list(frame().columns)
                ['id_left', 'text_left', 'id_right', 'text_right', 'label']
                >>> len(frame()) == len(data_pack)
                True

            Notice that a view is binded to the original data pack, so changing
            contents of the data pack will affect a view previously created:
                >>> data_pack.drop_label(inplace=True)
                >>> list(frame().columns)
                ['id_left', 'text_left', 'id_right', 'text_right']

            To slice the view:
                >>> frame_slice = frame[3:5]
                >>> len(frame_slice)
                2

            """
            self._data_pack = data_pack

        def __getitem__(self, index: typing.Union[int, slice, np.array]
                        ) -> pd.DataFrame:
            """Slicer."""
            dp = self._data_pack
            index = _convert_to_list_index(index, len(dp))
            left_df = dp.left.loc[dp.relation['id_left'][index]].reset_index()
            right_df = dp.right.loc[
                dp.relation['id_right'][index]].reset_index()
            joined_table = left_df.join(right_df)
            for column in dp.relation.columns:
                if column not in ['id_left', 'id_right']:
                    labels = dp.relation[column][index].to_frame()
                    labels = labels.reset_index(drop=True)
                    joined_table = joined_table.join(labels)
            return joined_table

        def __call__(self):
            """:return: A full copy. Equivalant to `frame[:]`."""
            return self[:]


def load_data_pack(dirpath: typing.Union[str, Path]) -> DataPack:
    """
    Load a :class:`DataPack`. The reverse function of :meth:`save`.

    :param dirpath: directory path of the saved model.
    :return: a :class:`DataPack` instance.
    """
    dirpath = Path(dirpath)

    data_file_path = dirpath.joinpath(DataPack.DATA_FILENAME)
    dp = dill.load(open(data_file_path, 'rb'))

    return dp
