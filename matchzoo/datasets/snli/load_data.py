"""SNLI data loader."""

from pathlib import Path

import pandas as pd
import keras

import matchzoo

_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"


def load_data(stage='train', task='classification', target_label='entailment'):
    """
    Load SNLI data.

    :param stage: One of `train`, `dev`, and `test`. (default: `train`)
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance. (default: `ranking`)
    :param target_label: If `ranking`, chose one of `entailment`,
        `contradiction`, `neutral`, and `-` as the positive label.
        (default: `entailment`)

    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_root = _download_data()
    file_path = data_root.joinpath(f'snli_1.0_{stage}.txt')
    data_pack = _read_data(file_path)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        if target_label not in ['entailment', 'contradiction', 'neutral', '-']:
            raise ValueError
        binary = (data_pack.relation['label'] == target_label).astype(float)
        data_pack.relation['label'] = binary
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        classes = ['entailment', 'contradiction', 'neutral', '-']
        label = data_pack.relation['label'].apply(classes.index)
        data_pack.relation['label'] = label
        data_pack.one_hot_encode_label(num_classes=4, inplace=True)
        return data_pack, classes
    else:
        raise ValueError(f"{task} is not a valid task.")


def _download_data():
    ref_path = keras.utils.data_utils.get_file(
        'snli', _url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='snli'
    )
    return Path(ref_path).parent.joinpath('snli_1.0')


def _read_data(path):
    table = pd.read_table(path)
    df = pd.DataFrame({
        'text_left': table['sentence1'],
        'text_right': table['sentence2'],
        'label': table['gold_label']
    })
    return matchzoo.pack(df)
