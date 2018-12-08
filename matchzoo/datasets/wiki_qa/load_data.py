"""WikiQA data loader."""

from pathlib import Path

import keras

import matchzoo

_url = "https://download.microsoft.com/download/E/5/F/" \
       "E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip"


def load_data(stage='train', task='ranking'):
    """
    Load WikiQA data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_root = _download_data()
    file_path = data_root.joinpath(f'WikiQA-{stage}.tsv')
    data_pack = _read_data(file_path)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        label = data_pack.relation['label'].astype(int).apply(
            task.one_hot_encode)
        data_pack.relation['label'] = label
        return data_pack, [False, True]
    else:
        raise ValueError(f"{task} is not a valid task.")


def _download_data():
    ref_path = keras.utils.data_utils.get_file('wikiqa', _url, extract=True)
    return Path(ref_path).parent.joinpath('WikiQACorpus')


def _read_data(path):
    def scan_file():
        with open(path) as in_file:
            next(in_file)  # skip header
            for l in in_file:
                qid, q, _, _, did, d, label = l.strip().split('\t')
                yield qid, did, q, d, float(label)

    return matchzoo.pack(list(scan_file()))
