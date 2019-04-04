import pytest

from matchzoo import tasks


@pytest.mark.parametrize("task_type", [
    tasks.Ranking, tasks.Classification
])
def test_task_listings(task_type):
    assert task_type.list_available_losses()
    assert task_type.list_available_metrics()


@pytest.mark.parametrize("arg", [None, -1, 0, 1])
def test_classification_instantiation_failure(arg):
    with pytest.raises(Exception):
        tasks.Classification(num_classes=arg)


@pytest.mark.parametrize("arg", [2, 10, 2048])
def test_classification_num_classes(arg):
    task = tasks.Classification(num_classes=arg)
    assert task.num_classes == arg
