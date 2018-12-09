"""SNLI data loader."""

from pathlib import Path

import json
import keras

import matchzoo

_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"


def load_data(stage='train', task='ranking'):
    """
    Load SNLI data.

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
    file_path = data_root.joinpath(f'snli_1.0_{stage}.jsonl')
    data_pack = _read_data(file_path)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        binary = (data_pack.relation['label'] == 'entailment').astype(int)
        data_pack.relation['label'] = binary
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        classes = ['neutral', 'contradiction', 'entailment', '-']
        label = data_pack.relation['label'].apply(classes.index)
        data_pack.relation['label'] = label
        return data_pack, classes
    else:
        raise ValueError(f"{task} is not a valid task.")


def _download_data():
    ref_path = keras.utils.data_utils.get_file(
        'snli', _url, extract=True, cache_dir=matchzoo.USER_DATA_DIR)
    return Path(ref_path).parent.joinpath('snli_1.0')


def _read_data(path):
    def scan_file():
        left_ids = {}
        right_ids = {}
        with open(path) as in_file:
            for line in in_file:
                obj = json.loads(line)
                text_left, text_right = obj['sentence1'], obj['sentence2']
                label = obj['gold_label']
                if text_left not in left_ids:
                    left_ids[text_left] = 'TEXT_' + str(len(left_ids))
                if text_right not in right_ids:
                    right_ids[text_right] = 'HYPO_' + str(len(right_ids))
                id_left = left_ids[text_left]
                id_right = right_ids[text_right]
                yield id_left, id_right, text_left, text_right, label

    return matchzoo.pack(list(scan_file()))
