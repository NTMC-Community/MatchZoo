"""Quora Question Pairs data loader."""

import typing
from pathlib import Path

import pandas as pd
from tensorflow import keras

import matchzoo

_url = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence" \
       "-representations.appspot.com/o/data%2FQQP.zip?alt=media&" \
       "token=700c6acf-160d-4d89-81d1-de4191d02cb5"


def load_data(
    stage: str = 'train',
    task: str = 'classification',
    return_classes: bool = False,
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load QuoraQP data.

    :param path: `None` for download from quora, specific path for
        downloaded data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param return_classes: Whether return classes for classification task.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_root = _download_data()
    file_path = data_root.joinpath(f"{stage}.tsv")
    data_pack = _read_data(file_path, stage)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    elif task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        if stage != 'test':
            data_pack.one_hot_encode_label(num_classes=2, inplace=True)
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task.")


def _download_data():
    ref_path = keras.utils.get_file(
        'quora_qp', _url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='quora_qp'
    )
    return Path(ref_path).parent.joinpath('QQP')


def _read_data(path, stage):
    data = pd.read_csv(path, sep='\t', error_bad_lines=False)
    data = data.dropna(axis=0, how='any').reset_index(drop=True)
    if stage in ['train', 'dev']:
        df = pd.DataFrame({
            'id_left': data['qid1'],
            'id_right': data['qid2'],
            'text_left': data['question1'],
            'text_right': data['question2'],
            'label': data['is_duplicate'].astype(int)
        })
    else:
        df = pd.DataFrame({
            'text_left': data['question1'],
            'text_right': data['question2']
        })
    return matchzoo.pack(df)
