"""CQA-QL-16 data loader."""

import xml
import typing
from pathlib import Path

import keras
import pandas as pd

import matchzoo


_train_dev_url = "http://alt.qcri.org/semeval2016/task3/data/uploads/" \
                 "semeval2016-task3-cqa-ql-traindev-v3.2.zip"
_test_url = "http://alt.qcri.org/semeval2016/task3/data/uploads/" \
            "semeval2016_task3_test.zip"


def load_data(
    stage: str = 'train',
    task: str = 'classification',
    target_label: str = 'PerfectMatch',
    return_classes: bool = False,
    match_type: str = 'question',
    mode: str = 'both',
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load CQA-QL-16 data.

    :param stage: One of `train`, `dev`, and `test`.
        (default: `train`)
    :param task: Could be one of `ranking`, `classification` or instance
        of :class:`matchzoo.engine.BaseTask`. (default: `classification`)
    :param target_label: If `ranking`, choose one of classification
        label as the positive label. (default: `PerfectMatch`)
    :param return_classes: `True` to return classes for classification
        task, `False` otherwise.
    :param match_type: Matching text types. One of `question`,
        `answer`, and `external_answer`. (default: `question`)
    :param mode: Train data use method. One of `part1`, `part2`,
        and `both`. (default: `both`)

    :return: A DataPack unless `task` is `classification` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    if match_type not in ('question', 'answer', 'external_answer'):
        raise ValueError(f"{match_type} is not a valid method. Must be one of"
                         f" `question`, `answer`, `external_answer`.")

    if mode not in ('part1', 'part2', 'both'):
        raise ValueError(f"{mode} is not a valid method."
                         f"Must be one of `part1`, `part2`, `both`.")

    data_root = _download_data(stage)
    data_pack = _read_data(data_root, stage, match_type, mode)

    if task == 'ranking':
        if match_type in ('anwer', 'external_answer') and target_label not in [
                'Good', 'PotentiallyUseful', 'Bad']:
            raise ValueError(f"{target_label} is not a valid target label."
                             f"Must be one of `Good`, `PotentiallyUseful`,"
                             f" `Bad`.")
        elif match_type == 'question' and target_label not in [
                'PerfectMatch', 'Relevant', 'Irrelevant']:
            raise ValueError(f"{target_label} is not a valid target label."
                             f" Must be one of `PerfectMatch`, `Relevant`,"
                             f" `Irrelevant`.")
        binary = (data_pack.relation['label'] == target_label).astype(float)
        data_pack.relation['label'] = binary
        return data_pack
    elif task == 'classification':
        if match_type in ('answer', 'external_answer'):
            classes = ['Good', 'PotentiallyUseful', 'Bad']
        else:
            classes = ['PerfectMatch', 'Relevant', 'Irrelevant']
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


def _read_data(path, stage, match_type, mode='both'):
    if stage == 'train':
        if mode == 'part1':
            path = path.joinpath(
                'train/SemEval2016-Task3-CQA-QL-train-part1.xml')
            data = _load_data_by_type(path, match_type)
        elif mode == 'part2':
            path = path.joinpath(
                'train/SemEval2016-Task3-CQA-QL-train-part2.xml')
            data = _load_data_by_type(path, match_type)
        else:
            part1 = path.joinpath(
                'train/SemEval2016-Task3-CQA-QL-train-part1.xml')
            p1 = _load_data_by_type(part1, match_type)
            part2 = path.joinpath(
                'train/SemEval2016-Task3-CQA-QL-train-part1.xml')
            p2 = _load_data_by_type(part2, match_type)
            data = pd.concat([p1, p2], ignore_index=True)
        return matchzoo.pack(data)
    elif stage == 'dev':
        path = path.joinpath('dev/SemEval2016-Task3-CQA-QL-dev.xml')
        data = _load_data_by_type(path, match_type)
        return matchzoo.pack(data)
    else:
        path = path.joinpath('SemEval2016-Task3-CQA-QL-test.xml')
        data = _load_data_by_type(path, match_type)
        return matchzoo.pack(data)


def _load_data_by_type(path, match_type):
    if match_type == 'question':
        return _load_question(path)
    elif match_type == 'answer':
        return _load_answer(path)
    else:
        return _load_external_answer(path)


def _load_question(path):
    doc = xml.etree.ElementTree.parse(path)
    dataset = []
    for question in doc.iterfind('OrgQuestion'):
        qid = question.attrib['ORGQ_ID']
        query = question.findtext('OrgQBody')
        rel_question = question.find('Thread').find('RelQuestion')
        question = rel_question.findtext('RelQBody')
        question_id = rel_question.attrib['RELQ_ID']
        dataset.append([qid, question_id, query, question,
                        rel_question.attrib['RELQ_RELEVANCE2ORGQ']])
    df = pd.DataFrame(dataset, columns=[
        'id_left', 'id_right', 'text_left', 'text_right', 'label'])
    return df


def _load_answer(path):
    doc = xml.etree.ElementTree.parse(path)
    dataset = []
    for org_q in doc.iterfind('OrgQuestion'):
        for thread in org_q.iterfind('Thread'):
            ques = thread.find('RelQuestion')
            qid = ques.attrib['RELQ_ID']
            question = ques.findtext('RelQBody')
            for comment in thread.iterfind('RelComment'):
                aid = comment.attrib['RELC_ID']
                answer = comment.findtext('RelCText')
                dataset.append([qid, aid, question, answer,
                                comment.attrib['RELC_RELEVANCE2RELQ']])
    df = pd.DataFrame(dataset, columns=[
        'id_left', 'id_right', 'text_left', 'text_right', 'label'])
    return df


def _load_external_answer(path):
    doc = xml.etree.ElementTree.parse(path)
    dataset = []
    for question in doc.iterfind('OrgQuestion'):
        qid = question.attrib['ORGQ_ID']
        query = question.findtext('OrgQBody')
        thread = question.find('Thread')
        for comment in thread.iterfind('RelComment'):
            answer = comment.findtext('RelCText')
            aid = comment.attrib['RELC_ID']
            dataset.append([qid, aid, query, answer,
                            comment.attrib['RELC_RELEVANCE2ORGQ']])
    df = pd.DataFrame(dataset, columns=[
        'id_left', 'id_right', 'text_left', 'text_right', 'label'])
    return df
