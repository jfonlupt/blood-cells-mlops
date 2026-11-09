"""Abstraction de tâche : centralise tout ce qui dépendait de la string ``mode``.

Un ``TaskSpec`` est le seul objet transmis à ``build_model``, ``build_loss``,
``build_metrics`` et ``BloodCellDataset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from src.constants import (
    BARCELONA_KEY,
    BINARY_CRITICALITY_CLASSES,
    CRITICALITY_CLASSES,
    FINAL_CLASSES,
    MUNICH_KEY,
)


@dataclass(frozen=True)
class TaskSpec:
    """Décrit une tâche de classification (labels, dimension, loss par défaut)."""

    name: str
    label_column: str
    class_names: list[str]
    num_classes: int
    kind: Literal["binary", "multiclass"]
    label_dtype: torch.dtype
    target_shape: Literal["scalar", "vector1"]
    default_loss: Literal["cross_entropy", "bce"]


TASKS: dict[str, TaskSpec] = {
    "dataset": TaskSpec(
        name="dataset",
        label_column="label",
        class_names=[BARCELONA_KEY, MUNICH_KEY],
        num_classes=2,
        kind="binary",
        label_dtype=torch.float32,
        target_shape="vector1",
        default_loss="bce",
    ),
    "type": TaskSpec(
        name="type",
        label_column="label",
        class_names=list(FINAL_CLASSES),
        num_classes=len(FINAL_CLASSES),
        kind="multiclass",
        label_dtype=torch.long,
        target_shape="scalar",
        default_loss="cross_entropy",
    ),
    "criticality": TaskSpec(
        name="criticality",
        label_column="label",
        class_names=list(CRITICALITY_CLASSES),
        num_classes=len(CRITICALITY_CLASSES),
        kind="multiclass",
        label_dtype=torch.long,
        target_shape="scalar",
        default_loss="cross_entropy",
    ),
    "binary_criticality": TaskSpec(
        name="binary_criticality",
        label_column="label",
        class_names=list(BINARY_CRITICALITY_CLASSES),
        num_classes=len(BINARY_CRITICALITY_CLASSES),
        kind="multiclass",
        label_dtype=torch.long,
        target_shape="scalar",
        default_loss="cross_entropy",
    ),
}


def get_task(name: str) -> TaskSpec:
    """Renvoie le ``TaskSpec`` nommé, ``KeyError`` explicite sinon."""
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(
            f"tâche inconnue : {name!r}. Choisir parmi {sorted(TASKS)}."
        ) from None
