"""Construction du DataFrame d'images (chemins + labels dérivés)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.constants import (
    BINARY_CRITICALITY,
    CRITICALITY,
    DATASET_PATHS,
    MAPPING_DICT,
    MODE_TO_LABEL,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.config import DataConfig
    from src.tasks import TaskSpec


def make_update_data(task: TaskSpec) -> Callable[[pd.Series], tuple]:
    """Renvoie une fonction ``row -> (type, label, binary_criticality, criticality, type_dataset)``."""

    def update_data(row: pd.Series) -> tuple:
        type_ = MAPPING_DICT[row.dataset].get(row.original_type)
        if type_ is None:
            return (None, None, None, None, None)
        criticality = CRITICALITY[type_]
        binary_criticality = BINARY_CRITICALITY[type_]

        if task.name == "criticality":
            label_str = criticality
        elif task.name == "binary_criticality":
            label_str = binary_criticality
        elif task.name == "dataset":
            label_str = row.dataset
        else:
            label_str = type_

        label = MODE_TO_LABEL[task.name][label_str]
        return type_, label, binary_criticality, criticality, type_ + "_" + row.dataset

    return update_data


def get_images(data_dir: str | Path, dataset_name: str) -> list[tuple[str, str, str]]:
    """Liste ``(chemin, dataset, type_originel)`` pour les images mappables."""
    data_path = Path(data_dir)
    image_paths = (
        list(data_path.glob("**/*.jpg"))
        + list(data_path.glob("**/*.png"))
        + list(data_path.glob("**/*.tiff"))
    )
    labels = [p.stem.split("_")[0] for p in image_paths]
    return [
        (str(p), dataset_name, label)
        for p, label in zip(image_paths, labels)
        if label in MAPPING_DICT[dataset_name]
    ]


def create_dataframe(data_cfg: DataConfig, task: TaskSpec) -> pd.DataFrame:
    """DataFrame des images retenues, avec labels dérivés pour la tâche ``task``."""
    data: list[tuple[str, str, str]] = []
    for key in data_cfg.datasets:
        data.extend(get_images(DATASET_PATHS[key], key))
    df = pd.DataFrame(data, columns=["path", "dataset", "original_type"])

    update_data = make_update_data(task)
    df[["type", "label", "binary_criticality", "criticality", "type_dataset"]] = df.apply(
        update_data, axis=1, result_type="expand"
    )
    df = df.dropna().reset_index(drop=True)

    n = data_cfg.n_images
    if n > 0 and data_cfg.stratify != "no":
        per_group = n // df[data_cfg.stratify].nunique()
        return (
            df.groupby(data_cfg.stratify, group_keys=False)[df.columns]
            .apply(lambda g: g.sample(n=min(len(g), per_group)))
            .reset_index(drop=True)
        )
    if n > 0:
        return df.sample(n).reset_index(drop=True)
    return df
