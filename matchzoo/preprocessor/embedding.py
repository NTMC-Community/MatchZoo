"""Matchzoo toolkit for token embedding."""
import numpy as np
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


class Embedding():
    """
    Embedding class.

    TODO: https://github.com/dmlc/gluon-nlp/blob/
          5e65c4751e1be8920b1021822a988af77224902f/
          gluonnlp/embedding/token_embedding.py

    Examples:
        >>> embed = Embedding('tests/unit_test/data/embed_10.txt')
        >>> # Need term_index
        >>> embed.process({'G': 1, 'C': 2, 'D': 3, 'A': 4, '[PAD]': 0})
        >>> index_state = embed.index_state
        >>> index_state # doctest: +SKIP
        {4: 1, 2: 1, 3: 1, 1: 2, 0: 0}
        >>> embed_mat = embed.embed_mat
        >>> embed_mat # doctest: +SKIP
        array([[ 0.       ,  0.       ,  0.       ,  0.       ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.22785383, -0.09405118,  0.15446669,  0.20727136,  0.05943427,
             0.12673594,  0.02788511, -0.0433806 , -0.20548974,  0.24532762],
           [ 0.1       ,  0.2       ,  0.3       ,  0.4       ,  0.5       ,
             0.6       ,  0.7       ,  0.8       ,  0.9       ,  1.        ],
           [ 0.1       ,  0.2       ,  0.3       ,  0.4       ,  0.5       ,
             0.6       ,  0.7       ,  0.8       ,  0.9       ,  1.        ],
           [ 0.1       ,  0.2       ,  0.3       ,  0.4       ,  0.5       ,
             0.6       ,  0.7       ,  0.8       ,  0.9       ,  1.        ]])
        >>> embed_mat[0] # doctest: +SKIP
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    """

    def __init__(self, embed_file: str):
        """
        Class initialization.

        :param embed_file: the embeding input file path.
        """
        self.embed_file = embed_file
        self.embed_dim = 0
        self.index_state = {}
        self.embed_mat = None

    def process(self, term_index: dict):
        """Build a :np.array:`embed_mat` and a :dict:`index_state`."""
        n_word_in_dict = max(term_index.values()) + 1
        if n_word_in_dict != len(term_index):
            logger.warning("Some words are not shown in term_index({}). Total "
                           "number of words are {}.".format(len(term_index),
                                                            n_word_in_dict))

        # The number of word share with `term_index` and `embed_file`.
        n_word_share = 0

        with open(self.embed_file, 'rb') as embed_file_ptr:
            # Read the header of the embed file
            n_word_in_embed, self.embed_dim = \
                map(int, embed_file_ptr.readline().rstrip().split(b" "))

            self.embed_mat = np.random.uniform(-0.25, 0.25, (n_word_in_dict,
                                                             self.embed_dim))

            # Go through all embedding file
            for line in tqdm(embed_file_ptr):
                # Explicitly splitting on " " is important, so we don't
                # get rid of Unicode non-breaking spaces in the vectors.
                entries = line.rstrip().split(b" ")

                word, vec = entries[0], entries[1:]

                if self.embed_dim != len(vec):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, but "
                        "previously read vectors have {} dimensions. "
                        "All vectors must have the same number of "
                        "dimensions.".format(word, len(vec), self.embed_dim))

                try:
                    if isinstance(word, bytes):
                        word = word.decode('utf-8')
                except Exception as e:
                    logger.warning(
                        "Skipping non-UTF8 token {}".format(repr(word)))
                    continue

                if word in term_index:
                    index = term_index[word]
                    self.embed_mat[index] = \
                        np.array([float(x) for x in vec])
                    self.index_state[index] = 1
                    n_word_share += 1

            # init tht OOV word embeddings
            for word in term_index:
                index = term_index[word]
                if index not in self.index_state:
                    self.index_state[index] = 2

            self.index_state[0] = 0
            self.embed_mat[0] = np.zeros([self.embed_dim])
