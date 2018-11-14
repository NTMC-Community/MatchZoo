from pathlib import Path

from matchzoo import pack, embedding

CURR_DIR = Path(__file__).parent


def load_data(path, include_label):
    def scan_file():
        with open(path) as in_file:
            next(in_file)  # skip header
            for l in in_file:
                yield l.strip().split('\t')

    if include_label:
        data = [(*args, float(label)) for *args, label in scan_file()]
    else:
        data = list(scan_file())
    return pack(data)


def load_train_classify_data():
    path = CURR_DIR.joinpath('train_classify.txt')
    return load_data(path, include_label=True)


def load_test_classify_data():
    path = CURR_DIR.joinpath('test_classify.txt')
    return load_data(path, include_label=False)


def load_train_rank_data():
    path = CURR_DIR.joinpath('train_rank.txt')
    return load_data(path, include_label=True)


def load_test_rank_data():
    path = CURR_DIR.joinpath('test_rank.txt')
    return load_data(path, include_label=False)


def load_embedding():
    path = CURR_DIR.joinpath('embed_10.txt')
    return embedding.Embedding(path)
