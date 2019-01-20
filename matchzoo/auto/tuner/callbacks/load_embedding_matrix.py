import matchzoo as mz
from matchzoo.auto.tuner.callbacks.callback import Callback


class LoadEmbeddingMatrix(Callback):
    """
    Load a pre-trained embedding after the model is built.

    :param embedding_matrix: Embedding matrix to load.

    """

    def __init__(self, embedding_matrix):
        """Init."""
        self._embedding_matrix = embedding_matrix

    def on_build_end(self, tuner, model: mz.engine.BaseModel):
        """`on_build_end`."""
        model.load_embedding_matrix(self._embedding_matrix)
