import typing
from pathlib import Path

import pandas as pd

import matchzoo


def load_data(
    stage: str = 'train',
    task: str = 'ranking',
    return_classes: bool = False
) -> typing.Union[matchzoo.DataPack, typing.Tuple[matchzoo.DataPack, list]]:
    """
    Load WikiQA data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.

    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.

    Example:
        >>> import matchzoo as mz
        >>> stages = 'train', 'dev', 'test'
        >>> tasks = 'ranking', 'classification'
        >>> for stage in stages:
        ...     for task in tasks:
        ...         _ = mz.datasets.toy.load_data(stage, task)
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    path = Path(__file__).parent.joinpath(f'{stage}.csv')
    data_pack = matchzoo.pack(pd.read_csv(path, index_col=0))

    if isinstance(task, matchzoo.tasks.Ranking):
        data_pack.relation['label'] = \
            data_pack.relation['label'].astype('float32')
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        data_pack.relation['label'] = data_pack.relation['label'].astype(int)
        data_pack = data_pack.one_hot_encode_label(num_classes=2)
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")
