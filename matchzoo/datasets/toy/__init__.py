from pathlib import Path

from matchzoo import pack, embedding

DATA_ROOT = Path(__file__).parent


def load_data(path, include_label):
    def scan_file():
        with open(path, encoding='utf-8') as in_file:
            next(in_file)  # skip header
            for l in in_file:
                yield l.strip().split('\t')

    if include_label:
        data = [(*args, float(label)) for *args, label in scan_file()]
    else:
        data = list(scan_file())
    return pack(data)


def load_train_classify_data():
    path = DATA_ROOT.joinpath('train_classify.txt')
    return load_data(path, include_label=True)


def load_test_classify_data():
    path = DATA_ROOT.joinpath('test_classify.txt')
    return load_data(path, include_label=False)


def load_train_rank_data():
    path = DATA_ROOT.joinpath('train_rank.txt')
    return load_data(path, include_label=True)


def load_test_rank_data():
    path = DATA_ROOT.joinpath('test_rank.txt')
    return load_data(path, include_label=False)
