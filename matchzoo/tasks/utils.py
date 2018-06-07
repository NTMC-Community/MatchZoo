"""Task utilities."""

import typing

from matchzoo import engine
from . import Ranking
from . import Classification


def list_available_task_types() -> typing.List[typing.Type[engine.BaseTask]]:
    """Return a list of task type class objects."""
    return [engine.BaseTask, Ranking, Classification]
