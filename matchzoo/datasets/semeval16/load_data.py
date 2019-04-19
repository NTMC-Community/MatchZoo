"""SemEval 16 data loader."""

import typing
from pathlib import Path

import keras
import pandas as pd
from xml.etree.ElementTree import parse

import matchzoo


_train_dev_url = "http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016-task3-cqa-ql-traindev-v3.2.zip"
_test_url = "http://alt.qcri.org/semeval2016/task3/data/uploads/semeval2016_task3_test.zip"


def load_data(
    stage: str = 'train',
    pair_type: str = 'answer',
    method: str = 'both',
    task: str = 'classification',
    target_label: str = 'Good',
    return_classes: bool = False,
) -> typing.Union[matchzoo.DataPack, tuple]:
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    if method not in ('part1', 'part2', 'both'):
        raise ValueError(f"{method} is not a valid method."
                         f"Must be one of `part1`, `part2`, `both`.")

    if pair_type not in ('question', 'answer', 'external_answer'):
        raise ValueError(f"{method} is not a valid method."
                         f"Must be one of `part1`, `part2`, `both`.")

    data_root = _download_data(stage)
    data_pack = _read_data(data_root, stage, pair_type)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        if target_label not in ['Good', 'PotentiallyUseful', 'Bad']:
            raise ValueError(f"{target_label} is not a valid target label."
                             f"Must be one of `Good`, `PotentiallyUseful`, `Bad`.")
        binary = (data_pack.relation['label'] == target_label).astype(float)
        data_pack.relation['label'] = binary
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        classes = ['Good', 'PotentiallyUseful', 'Bad']
        label = data_pack.relation['label'].apply(classes.index)
        data_pack.relation['label'] = label
        data_pack.one_hot_encode_label(num_classes=3, inplace=True)
        if return_classes:
            return data_pack, classes
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")


def _download_data(stage):
    if stage in ['train', 'dev']:
        return _download_train_dev_data()
    else:
        return _download_test_data()


def _download_train_dev_data():
    ref_path = keras.utils.data_utils.get_file(
        'semeval_train', _train_dev_url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='semeval_train'
    )
    return Path(ref_path).parent.joinpath('v3.2')


def _download_test_data():
    ref_path = keras.utils.data_utils.get_file(
        'semeval_test', _test_url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='semeval_test'
    )
    return Path(ref_path).parent.joinpath('SemEval2016_task3_test/English')


def _read_data(path, stage, pair_type, method='both'):
    if stage == 'train':
        if method == 'part1':
            path = path.joinpath('train/SemEval2016-Task3-CQA-QL-train-part1.xml')
            data = _load_data_by_type(path, pair_type, True)
        elif method == 'part2':
            path = path.joinpath('train/SemEval2016-Task3-CQA-QL-train-part2.xml')
            data = _load_data_by_type(path, pair_type, True)
        else:
            part1 = path.joinpath('train/SemEval2016-Task3-CQA-QL-train-part1.xml')
            p1 = _load_data_by_type(part1, pair_type, True)
            part2 = path.joinpath('train/SemEval2016-Task3-CQA-QL-train-part1.xml')
            p2 = _load_data_by_type(part2, pair_type, True)
            data = pd.concate([p1, p2], ignore_index=True)
        return matchzoo.pack(data)
    elif stage == 'dev':
        path = path.joinpath('dev/SemEval2016-Task3-CQA-QL-dev.xml')
        data = _load_data_by_type(path, pair_type, True)
        return matchzoo.pack(data)
    else:
        path = path.joinpath('SemEval2016-Task3-CQA-QL-test.xml')
        data = _load_data_by_type(path, pair_type, False)
        return matchzoo.pack(data)


def _load_data_by_type(path, pair_type, has_label):
    if pair_type == 'question':
        return _load_question(path, has_label)
    elif pair_type == 'answer':
        return _load_answer(path, has_label)
    else:
        return _load_external_answer(path, has_label)


def _load_question(path, has_label):
    doc = parse(path)
    dataset = []
    for question in doc.iterfind('OrgQuestion'):
        qid = question.attrib['ORGQ_ID']
        query = question.findtext('OrgQBody')
        rel_question = question.find('Thread').find('RelQuestion')
        question = rel_question.findtext('RelQBody')
        question_id = rel_question.attrib['RELQ_ID']
        sample = [qid, question_id, query, question]
        if has_label:
            sample.append(rel_question.attrib['RELQ_RELEVANCE2ORGQ'])
        dataset.append(sample)
    if has_label:
        df = pd.DataFrame(dataset, columns=['id_left', 'id_right', 'text_left', 'text_right', 'label'])
    else:
        df = pd.DataFrame(dataset, columns=['id_left', 'id_right', 'text_left', 'text_right'])
    return df


def _load_answer(path, has_label):
    doc = parse(path)
    dataset = []
    for thread in doc.iterfind('Thread'):
        ques = thread.find('RelQuestion')
        qid = ques.attrib['RELQ_ID']
        question = ques.findtext('RelQBody')
        for comment in thread.iterfind('RelComment'):
            aid = comment.attrib['RELC_ID']
            answer = comment.findtext['RelCText']
            sample = [qid, aid, question, answer]
            if has_label:
                sample.append(comment.attrib['RELC_RELEVANCE2RELQ'])
            dataset.append(sample)
    if has_label:
        df = pd.DataFrame(dataset, columns=['id_left', 'id_right', 'text_left', 'text_right', 'label'])
    else:
        df = pd.DataFrame(dataset, columns=['id_left', 'id_right', 'text_left', 'text_right'])
    return df


def _load_external_answer(path, has_label=False):
    doc = parse(path)
    dataset = []
    for question in doc.iterfind('OrgQuestion'):
        qid = question.attrib['ORGQ_ID']
        query = question.findtext('OrgQBody')
        thread = question.find('Thread')
        for comment in thread.iterfind('RelComment'):
            answer = comment.findtext('RelCText')
            aid = comment.attrib['RELC_ID']
            sample = [qid, aid, query, answer]
            if has_label:
                sample.append(comment.attrib['RELC_RELEVANCE2ORGQ'])
            dataset.append(sample)
    if has_label:
        df = pd.DataFrame(dataset, columns=['id_left', 'id_right', 'text_left', 'text_right', 'label'])
    else:
        df = pd.DataFrame(dataset, columns=['id_left', 'id_right', 'text_left', 'text_right'])
    return df




















