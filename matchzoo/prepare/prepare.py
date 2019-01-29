import typing

import matchzoo as mz
from .preparer import Preparer
from matchzoo.engine.base_task import BaseTask
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_preprocessor import BasePreprocessor


def prepare(
    task: BaseTask,
    model_class: typing.Type[BaseModel],
    data_pack: mz.DataPack,
    config: typing.Optional[dict] = None,
    preprocessor: typing.Optional[BasePreprocessor] = None,
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
