import typing

import numpy as np

import matchzoo as mz
from matchzoo.data_generator import DataGeneratorBuilder


class ZooKeeper(object):
    """
    Zoo Keeper. Unifies setup processes of all MatchZoo models.

    `config` is used to control specific behaviors. The default `config`
    will be updated accordingly if a `config` dictionary is passed. e.g. to
    override the default `bin_size`, pass `config={'bin_size': 15}`.

    See `tutorials/automation.ipynb` for a detailed walkthrough on usage.

    Default `config`:

    {
        # pair generator builder kwargs
        'num_dup': 1,

        # histogram unit of DRMM
        'bin_size': 30,
        'hist_mode': 'LCH',

        # dynamic Pooling of MatchPyramid
        'compress_ratio_left': 1.0,
        'compress_ratio_right': 1.0,

        # if no `matchzoo.Embedding` is passed to `tune`
        'embedding_output_dim': 50
    }

    :param task: Task.
    :param config: Configuration of specific behaviors.

    Example:
        >>> import matchzoo as mz
        >>> task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss())
        >>> keeper = mz.ZooKeeper(task)
        >>> model_class = mz.models.DenseBaseline
        >>> train_raw = mz.datasets.toy.load_data('train', 'ranking')
        >>> model, prpr, gen_builder, matrix = keeper.prepare(model_class,
        ...                                                   train_raw)
        >>> model.params.completed()
        True

    """

    def __init__(
        self,
        task: mz.engine.BaseTask,
        config: typing.Optional[dict] = None
    ):
        """Init."""
        self._task = task
        self._config = self._get_default_config()
        if config:
            self._config.update(config)

        self._infer_num_neg()

    def prepare(
        self,
        model_class: typing.Type[mz.engine.BaseModel],
        data_pack: mz.DataPack,
        preprocessor: typing.Optional[mz.engine.BasePreprocessor] = None,
        embedding: typing.Optional[mz.Embedding] = None,
    ) -> typing.Tuple[
        mz.engine.BaseModel,
        mz.engine.BasePreprocessor,
        DataGeneratorBuilder,
        np.ndarray
    ]:
        """
        Prepare.

        :param model_class: Model class.
        :param data_pack: DataPack used to fit the preprocessor.
        :param preprocessor: Preprocessor used to fit the `data_pack`.
            (default: the default preprocessor of `model_class`)
        :param embedding: Embedding to build a embedding matrix. If not set,
            then a correctly shaped randomized matrix will be built.

        :return: A tuple of `(model, preprocessor, data_generator_builder,
            embedding_matrix)`.

        """
        if not preprocessor:
            preprocessor = model_class.get_default_preprocessor()
        preprocessor.fit(data_pack, verbose=0)

        model = self._setup_model(model_class, preprocessor)

        if 'with_embedding' in model.params:
            embedding_matrix = self._build_matrix(preprocessor, embedding)
            model.params['embedding_input_dim'] = embedding_matrix.shape[0]
            model.params['embedding_output_dim'] = embedding_matrix.shape[1]
        else:
            embedding_matrix = None

        model.build()
        model.compile()

        builder_kwargs = self._infer_builder_kwargs(
            model, embedding_matrix)

        return (
            model,
            preprocessor,
            DataGeneratorBuilder(**builder_kwargs),
            embedding_matrix
        )

    def _setup_model(self, model_class, preprocessor):
        model = model_class()
        model.params['task'] = self._task
        model.params.update(preprocessor.context)
        if 'with_embedding' in model.params:
            model.params.update({
                'embedding_input_dim': preprocessor.context['vocab_size']
            })
        self._handle_drmm_input_shapes(model)
        return model

    def _handle_drmm_input_shapes(self, model):
        if isinstance(model, mz.models.DRMM):
            left = model.params['input_shapes'][0]
            orig_right = model.params['input_shapes'][1]
            updated_right = orig_right + self._config['bin_size']
            model.params['input_shapes'] = (left, updated_right)

    def _build_matrix(self, preprocessor, embedding):
        if embedding:
            vocab_unit = preprocessor.context['vocab_unit']
            term_index = vocab_unit.state['term_index']
            embedding_matrix = embedding.build_matrix(term_index)
            return embedding_matrix
        else:
            matrix_shape = (
                preprocessor.context['vocab_size'],
                self._config['embedding_output_dim']
            )
            return np.random.uniform(-0.2, 0.2, matrix_shape)

    def _infer_builder_kwargs(self, model, embedding_matrix):
        builder_kwargs = dict()
        if isinstance(self._task.loss, (mz.losses.RankHingeLoss,
                                        mz.losses.RankCrossEntropyLoss)):
            builder_kwargs.update(dict(
                pair_wise=True,
                num_dup=self._config['num_dup'],
                num_neg=self._config['num_neg']
            ))
        if isinstance(model, mz.models.DRMM):
            histo_callback = mz.data_generator.callbacks.Histogram(
                embedding_matrix=embedding_matrix,
                bin_size=self._config['bin_size'],
                hist_mode=self._config['hist_mode']
            )
            builder_kwargs['callbacks'] = [histo_callback]
        elif isinstance(model, mz.models.MatchPyramid):
            dpool_callback = mz.data_generator.callbacks.DynamicPooling(
                fixed_length_left=model.params['input_shapes'][0][0],
                fixed_length_right=model.params['input_shapes'][1][0],
                compress_ratio_left=self._config['compress_ratio_left'],
                compress_ratio_right=self._config['compress_ratio_right']
            )
            builder_kwargs['callbacks'] = [dpool_callback]
        return builder_kwargs

    def _infer_num_neg(self):
        if isinstance(self._task.loss, (mz.losses.RankHingeLoss,
                                        mz.losses.RankCrossEntropyLoss)):
            self._config['num_neg'] = self._task.loss.num_neg

    @classmethod
    def _get_default_config(cls):
        return {
            # pair generator builder kwargs
            'num_dup': 1,

            # histogram unit of DRMM
            'bin_size': 30,
            'hist_mode': 'LCH',

            # dynamic Pooling of MatchPyramid
            'compress_ratio_left': 1.0,
            'compress_ratio_right': 1.0,

            # if no `matchzoo.Embedding` is passed to `tune`
            'embedding_output_dim': 100
        }
