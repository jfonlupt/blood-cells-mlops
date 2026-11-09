"""Tests de l'abstraction de tâche."""

import pytest
import torch

from src.tasks import TASKS, get_task


def test_all_four_tasks_registered():
    assert set(TASKS) == {"dataset", "type", "criticality", "binary_criticality"}


@pytest.mark.parametrize(
    ("name", "num_classes", "kind", "dtype", "shape"),
    [
        ("dataset", 2, "binary", torch.float32, "vector1"),
        ("type", 10, "multiclass", torch.long, "scalar"),
        ("criticality", 3, "multiclass", torch.long, "scalar"),
        ("binary_criticality", 2, "multiclass", torch.long, "scalar"),
    ],
)
def test_task_shapes(name, num_classes, kind, dtype, shape):
    task = get_task(name)
    assert task.num_classes == num_classes
    assert task.kind == kind
    assert task.label_dtype == dtype
    assert task.target_shape == shape
    assert len(task.class_names) == num_classes


def test_get_task_unknown_raises():
    with pytest.raises(KeyError, match="inconnue"):
        get_task("cell_type")
