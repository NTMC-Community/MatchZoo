# coding: utf8
from __future__ import print_function

import sys
import six
import io
import array
import numpy as np
from tqdm import tqdm
import pickle


def load_word_embedding(vocab, w2v_file):
    """
    Pros:
        Save the oov words in oov.p for further analysis.
    Refs:
        class Vectors, https://github.com/pytorch/text/blob/master/torchtext/vocab.py
    Args:
        vocab: dict,
        w2v_file: file, path to file of pre-trained word2vec/glove/fasttext
    Returns:
        vectors
    """

    pre_trained = {}
    n_words = len(vocab)
    embeddings = None  # (n_words, n_dim)

    # str call is necessary for Python 2/3 compatibility, since
    # argument must be Python 2 str (Python 3 bytes) or
    # Python 3 str (Python 2 unicode)
    vectors, dim = array.array(str('d')), None

    # Try to read the whole file with utf-8 encoding.
    binary_lines = False
    try:
        with io.open(w2v_file, encoding="utf8") as f:
            lines = [line for line in f]
    # If there are malformed lines, read in binary mode
    # and manually decode each word from utf-8
    except:
        print("Could not read {} as UTF8 file, "
              "reading file as bytes and skipping "
              "words with malformed UTF8.".format(w2v_file))
        with open(w2v_file, 'rb') as f:
            lines = [line for line in f]
        binary_lines = True

    print("Loading vectors from {}".format(w2v_file))

    for line in tqdm(lines):
        # Explicitly splitting on " " is important, so we don't
        # get rid of Unicode non-breaking spaces in the vectors.
        entries = line.rstrip().split(b" " if binary_lines else " ")

        word, entries = entries[0], entries[1:]
        if dim is None and len(entries) > 1:
            dim = len(entries)
            # init the embeddings
            embeddings = np.random.uniform(-0.25, 0.25, (n_words, dim))

        elif len(entries) == 1:
            print("Skipping token {} with 1-dimensional "
                  "vector {}; likely a header".format(word, entries))
            continue
        elif dim != len(entries):
            raise RuntimeError(
                "Vector for token {} has {} dimensions, but previously "
                "read vectors have {} dimensions. All vectors must have "
                "the same number of dimensions.".format(word, len(entries), dim))

        if binary_lines:
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')

            except:
                print("Skipping non-UTF8 token {}".format(repr(word)))
                continue

        if word in vocab and word not in pre_trained:
            embeddings[vocab[word]] = [float(x) for x in entries]
            pre_trained[word] = 1

    pre_trained_len = len(pre_trained)
    print('Pre-trained: {}/{} {:.2f}'.format(pre_trained_len, n_words, pre_trained_len * 100.0 / n_words))

    oov_word_list = [w for w in vocab if w not in pre_trained]
    print('oov word list example (30): ', oov_word_list[:30])
    pickle.dump(oov_word_list, open('oov.p', 'wb'), protocol=2)

    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings


def load_word_dict(word_map_file):
    """ file -> {word: index} """
    word_dict = {}
    for line in tqdm(io.open(word_map_file, encoding='utf8')):
        line = line.split()
        try:
            word_dict[line[0]] = int(line[1])
        except:
            print(line)
            continue
    return word_dict


if __name__ == '__main__':

    w2v_file = sys.argv[1]  # w2v_file
    word_dict_file = sys.argv[2]  # word_dict_file
    mapped_w2v_file = sys.argv[3] # output shared w2v dict

    word_dict = {}

    print('load word dict ...')
    word_dict = load_word_dict(word_dict_file)

    print('load word vectors ...')
    embeddings = load_word_embedding(word_dict, w2v_file)

    print('save word vectors ...')
    with open(mapped_w2v_file, 'w') as fw:
        # assert word_dict
        for w, idx in tqdm(word_dict.items()):
            print(word_dict[w], ' '.join(map(str, embeddings[idx])), file=fw)

    print('Map word vectors finished ...')

