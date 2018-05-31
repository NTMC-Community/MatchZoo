import typing
import inspect

from matchzoo import engine
from . import Ranking
from . import BinaryClassification


def list_available_task_types() -> typing.List[typing.Type[engine.BaseTask]]:
    return [engine.BaseTask, Ranking, BinaryClassification]


def get(
        id_: typing.Union[str, engine.BaseTask, typing.Type[engine.BaseTask]]
) -> typing.Type[engine.BaseTask]:
    if inspect.isclass(id_) and issubclass(id_, engine.BaseTask):
        return id_
    elif id_ == 'ranking':
        return Ranking
    elif id_ == 'binary_classification' or id_ == 'classification':
        return BinaryClassification
    else:
        raise TypeError(f"Identifier {id_} Can not be intepreted.")
