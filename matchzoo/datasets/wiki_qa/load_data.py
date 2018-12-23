"""WikiQA data loader."""

import csv
from pathlib import Path

import keras
import pandas as pd

import matchzoo

_url = "https://download.microsoft.com/download/E/5/F/" \
       "E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip"


def load_data(stage='train', task='ranking', filter=False):
    """
    Load WikiQA data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filter: Whether remove the questions without correct answers.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.

    Examples:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.wiki_qa.load_data('train')
        >>> len(train_data)
        20360
        >>> dev_data = mz.datasets.wiki_qa.load_data('dev')
        >>> len(dev_data)
        2733
        >>> dev_data = mz.datasets.wiki_qa.load_data('dev', filter=True)
        >>> len(dev_data)
        1126
        >>> test_data = mz.datasets.wiki_qa.load_data('test')
        >>> len(test_data)
        6165
        >>> test_data = mz.datasets.wiki_qa.load_data('test', filter=True)
        >>> len(test_data)
        2341

    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_root = _download_data()
    file_path = data_root.joinpath(f'WikiQA-{stage}.tsv')
    data_pack = _read_data(file_path)
    if filter and stage in ('dev', 'test'):
        ref_path = data_root.joinpath(f'WikiQA-{stage}.ref')
        filter_ref_path = data_root.joinpath(f'WikiQA-{stage}-filtered.ref')
        filtered_ids = set([line.split()[0] for line in open(filter_ref_path)])
        filtered_lines = [idx for idx, line in enumerate(open(ref_path)) if
                          line.split()[0] in filtered_ids]
        data_pack = data_pack[filtered_lines]

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        data_pack.one_hot_encode_label(task.num_classes, inplace=True)
        return data_pack, [False, True]
    else:
        raise ValueError(f"{task} is not a valid task.")


def _download_data():
    ref_path = keras.utils.data_utils.get_file(
        'wikiqa', _url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='wiki_qa'
    )
    return Path(ref_path).parent.joinpath('WikiQACorpus')


def _read_data(path):
    table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    df = pd.DataFrame({
        'text_left': table['Question'],
        'text_right': table['Sentence'],
        'id_left': table['QuestionID'],
        'id_right': table['SentenceID'],
        'label': table['Label']
    })
    return matchzoo.pack(df)
