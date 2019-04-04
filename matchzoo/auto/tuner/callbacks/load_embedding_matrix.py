from matchzoo.engine.base_model import BaseModel
from .callback import Callback


class LoadEmbeddingMatrix(Callback):
    """
    Load a pre-trained embedding after the model is built.

    :param embedding_matrix: Embedding matrix to load.

    """

    def __init__(self, embedding_matrix):
        """Init."""
        self._embedding_matrix = embedding_matrix

    def on_build_end(self, tuner, model: BaseModel):
        """`on_build_end`."""
        model.load_embedding_matrix(self._embedding_matrix)
