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
        >>> embedding_mat = embed.embedding_mat
        >>> embedding_mat # doctest: +SKIP
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
        >>> embedding_mat[0] # doctest: +SKIP
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    """

    def __init__(self, embedding_file: str):
        """
        Class initialization.

        :param embedding_file: the embeding input file path.
        """
        self._embedding_file = embedding_file
        self._dim_embedding = 0
        self._index_state = {}
        self._embedding_mat = None

    @property
    def embedding_file(self):
        """Get embedding file name."""
        return self._embedding_file

    @property
    def dim_embedding(self):
        """Get embedding dimension."""
        return self._dim_embedding

    @property
    def index_state(self):
        """
        Get word index state dictionary.

        0 - [PAD] word, 1 - found embed, 2 - OOV word.
        """
        return self._index_state

    @property
    def embedding_mat(self):
        """Get constructed embedding matrix."""
        return self._embedding_mat

    def process(self, term_index: dict):
        """Build a :attr:`embedding_mat` and a :attr:`index_state`."""
        num_dict_words = max(term_index.values()) + 1
        if num_dict_words != len(term_index):
            logger.warning("Some words are not shown in term_index({}). Total "
                           "number of words are {}.".format(len(term_index),
                                                            num_dict_words))

        # The number of word share with `term_index` and `embedding_file`.
        num_shared_words = 0

        with open(self._embedding_file, 'rb') as embedding_file_ptr:
            # Read the header of the embed file
            num_embed_words, self._dim_embedding = \
                map(int, embedding_file_ptr.readline().rstrip().split(b" "))

            self._embedding_mat = np.random.uniform(-0.25, 0.25,
                                                    (num_dict_words,
                                                     self._dim_embedding))

            # Go through all embedding file
            for line in tqdm(embedding_file_ptr):
                # Explicitly splitting on " " is important, so we don't
                # get rid of Unicode non-breaking spaces in the vectors.
                entries = line.rstrip().split(b" ")

                word, vec = entries[0], entries[1:]

                if self._dim_embedding != len(vec):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, but "
                        "previously read vectors have {} dimensions. "
                        "All vectors must have the same number of "
                        "dimensions.".format(word, len(vec),
                                             self._dim_embedding))

                try:
                    if isinstance(word, bytes):
                        word = word.decode('utf-8')
                except Exception as e:
                    logger.warning(
                        "Skipping non-UTF8 token {}".format(repr(word)))
                    continue

                index = term_index.get(word, None)
                if index:
                    self._embedding_mat[index] = np.array(vec).astype(float)
                    self._index_state[index] = 1
                    num_shared_words += 1

            # init tht OOV word embeddings
            for word in term_index:
                index = term_index[word]
                if index not in self._index_state:
                    self._index_state[index] = 2

            self._index_state[0] = 0
            self._embedding_mat[0] = np.zeros([self._dim_embedding])
