import typing

import matchzoo as mz
from .preparer import Preparer


def prepare(
    task: mz.engine.BaseTask,
    model_class: typing.Type[mz.engine.BaseModel],
    data_pack: mz.DataPack,
    config: typing.Optional[dict] = None,
    preprocessor: typing.Optional[mz.engine.BasePreprocessor] = None,
    embedding: typing.Optional[mz.Embedding] = None,
):
    """
    A simple shorthand for using :class:`matchzoo.Preparer`.
    :param task:
    :param model_class:
    :param data_pack:
    :param config:
    :param preprocessor:
    :param embedding:
    :return:
    """
    preparer = Preparer(task=task, config=config)
    return preparer.prepare(
        model_class=model_class,
        data_pack=data_pack,
        preprocessor=preprocessor,
        embedding=embedding
    )
