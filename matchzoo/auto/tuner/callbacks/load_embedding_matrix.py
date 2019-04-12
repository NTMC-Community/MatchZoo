from matchzoo.engine.base_model import BaseModel
from matchzoo.auto.tuner.callbacks.callback import Callback


class LoadEmbeddingMatrix(Callback):
    """
    Load a pre-trained embedding after the model is built.

    Used with tuner to load a pre-trained embedding matrix for each newly built
    model instance.

    :param embedding_matrix: Embedding matrix to load.

    Example:

        >>> import matchzoo as mz
        >>> model = mz.models.ArcI()
        >>> prpr = model.get_default_preprocessor()
        >>> data = mz.datasets.toy.load_data()
        >>> data = prpr.fit_transform(data, verbose=0)
        >>> embed = mz.datasets.toy.load_embedding()
        >>> term_index = prpr.context['vocab_unit'].state['term_index']
        >>> matrix = embed.build_matrix(term_index)
        >>> callback = mz.auto.tuner.callbacks.LoadEmbeddingMatrix(matrix)
        >>> model.params.update(prpr.context)
        >>> model.params['task'] = mz.tasks.Ranking()
        >>> model.params['embedding_output_dim'] = embed.output_dim
        >>> result = mz.auto.tune(
        ...     params=model.params,
        ...     train_data=data,
        ...     test_data=data,
        ...     num_runs=1,
        ...     callbacks=[callback],
        ...     verbose=0
        ... )

    """

    def __init__(self, embedding_matrix):
        """Init."""
        self._embedding_matrix = embedding_matrix

    def on_build_end(self, tuner, model: BaseModel):
        """`on_build_end`."""
        model.load_embedding_matrix(self._embedding_matrix)
