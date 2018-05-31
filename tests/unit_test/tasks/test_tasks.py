import pytest

from matchzoo import tasks


def test_list_available_task_types():
    assert tasks.list_available_task_types()


@pytest.mark.parametrize("task_type", [
    tasks.Ranking, tasks.BinaryClassification
])
def test_task_listings(task_type):
    assert task_type.list_available_losses()
    assert task_type.list_available_metrics()


@pytest.mark.parametrize("id_, expected_task_type", [
    (tasks.BinaryClassification, tasks.BinaryClassification),
    ('classification', tasks.BinaryClassification),
    ('binary_classification', tasks.BinaryClassification),

    (tasks.Ranking, tasks.Ranking),
    ('ranking', tasks.Ranking),
])
def test_get_task_instance(id_, expected_task_type):
    assert tasks.get(id_) is expected_task_type


@pytest.mark.parametrize("id_", [
    None,
    'some_random_text'
])
def test_get_task_instance_failures(id_):
    with pytest.raises(TypeError):
        tasks.get(id_)
