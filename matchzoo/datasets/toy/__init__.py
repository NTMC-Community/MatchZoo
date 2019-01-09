from pathlib import Path

import pandas as pd

import matchzoo


def load_data(stage='train', task='ranking'):
    """
    Load WikiQA data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.

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
        return data_pack.one_hot_encode_label(num_classes=2), [False, True]
    else:
        raise ValueError(f"{task} is not a valid task.")
