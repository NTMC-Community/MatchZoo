"""Task types definitions."""

import typing


class BaseTask(object):
    """Base Task, shouldn't be used directly."""


class Ranking(BaseTask):
    """Ranking Task."""


class Classification(BaseTask):
    """Classification Task."""


def list_available_task_types() -> typing.List[typing.Type[BaseTask]]:
    return [BaseTask, Ranking, Classification]


def get(id_: typing.Union[str, BaseTask, typing.Type[BaseTask]]) -> BaseTask:
    if isinstance(id_, BaseTask):
        return id_
    elif isinstance(id_, type) and issubclass(id_, BaseTask):
        return id_()
    elif id_ == 'base' or id_ == 'base_task':
        return BaseTask()
    elif id_ == 'ranking':
        return Ranking()
    elif id_ == 'classification':
        return Classification()
    else:
        raise ValueError(f"Identifier {id_} Can not be intepreted.")
