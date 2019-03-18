"""Quora Question Pairs data loader."""

import typing
from pathlib import Path

import pandas as pd
import tensorflow as tf

import matchzoo

_url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"


def load_data(
    path: typing.Union[str, Path, None] = None,
    stage: str = 'train', task: str = 'classification'
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load QuoraQP data.

    :param path: `None` for download from quora, specific path for
        downloaded data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    if path is None:
        path = _download_data()
        dp = _read_data(path)
    else:
        if stage in ('train', 'dev'):
            dp = _read_data(path)
        else:
            dp = _read_data(path, False)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    elif task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return dp
    elif isinstance(task, matchzoo.tasks.Classification):
        dp.one_hot_encode_label(num_classes=2, inplace=True)
        return dp, [False, True]
    else:
        raise ValueError(f"{task} is not a valid task.")


def _download_data():
    ref_path = tf.keras.utils.get_file(
        'quoraqp', _url, extract=False,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='quora_qp'
    )
    return Path(ref_path).parent.joinpath('quora_duplicate_questions.tsv')


def _read_data(path, has_label=True):
    data = pd.read_csv(path)
    if has_label:
        df = pd.DataFrame({
            'id_left': data['qid1'],
            'id_right': data['qid2'],
            'text_left': data['question1'],
            'text_right': data['question2'],
            'label': data['is_duplicate']
        })
    else:
        df = pd.DataFrame({
            'text_left': data['question1'],
            'text_right': data['question2'],
            'label': pd.Series([0 for _ in range(len(data))])
        })
    return matchzoo.pack(df)
