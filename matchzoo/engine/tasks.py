"""Task types definitions."""


class BaseTask(object):
    """Base Task, shouldn't be used directly."""


class Ranking(BaseTask):
    """Ranking Task."""


class Classification(BaseTask):
    """Classification Task."""


def list_avaliable_task_types():
    return [BaseTask, Ranking, Classification]


def get(target):
    if issubclass(target, BaseTask):
        return target()
    elif isinstance(target, BaseTask):
        return target
    elif target == 'base':
        return BaseTask()
    elif target == 'ranking':
        return Ranking()
    elif target == 'classification':
        return Classification()
    else:
        raise NotImplementedError
