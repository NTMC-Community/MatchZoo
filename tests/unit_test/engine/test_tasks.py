import pytest
from matchzoo.engine import tasks


def test_list_available_task_types():
    assert tasks.list_available_task_types()


@pytest.mark.parametrize("task_type", tasks.list_available_task_types())
def test_task_instanctiation(task_type):
    assert task_type()


@pytest.mark.parametrize("id_, expected_task_type", [
    (tasks.BaseTask, tasks.BaseTask),
    (tasks.BaseTask(), tasks.BaseTask),
    ('base', tasks.BaseTask),
    ('base_task', tasks.BaseTask),

    (tasks.Classification, tasks.Classification),
    (tasks.Classification(), tasks.Classification),
    ('classification', tasks.Classification),

    (tasks.Ranking, tasks.Ranking),
    (tasks.Ranking(), tasks.Ranking),
    ('ranking', tasks.Ranking),
])
def test_get_task_instance(id_, expected_task_type):
    assert isinstance(tasks.get(id_), expected_task_type)


@pytest.mark.parametrize("id_", [
    'some_random_text',
    None
])
def test_get_task_instance_failures(id_):
    with pytest.raises(ValueError):
        tasks.get(id_)
