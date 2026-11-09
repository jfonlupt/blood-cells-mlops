# Refonte du pipeline d'entraînement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer `src/train.py` (monolithe argparse) et `src/models/base_model.py` (boucle mélangée au `nn.Module`) par une architecture config-driven testable, avec un `Trainer` isolé, une abstraction de tâche, et un grid search séquentiel produisant un dossier par run + un `leaderboard.csv`.

**Architecture :** Une `ExperimentConfig` pydantic (arbre de sous-configs typées, sans import torch) est chargée depuis un YAML + overrides `--set`. Un `runner` en tire un `TaskSpec`, un DataFrame, des splits, des transforms (stats calées sur le train), instancie un `Trainer` par fold, entraîne, évalue sur le test, et écrit les artefacts. Un `grid` déplie une grille de surcharges (`sklearn.model_selection.ParameterGrid`) en configs, lance chaque run, et agrège une ligne par run dans `leaderboard.csv`. Les modèles redeviennent de purs `nn.Module` ; les constructeurs sont des fonctions `build_*` libres à côté de leur domaine.

**Tech Stack :** Python 3.14, PyTorch 2.13, torchvision v2, torchmetrics, torchstain, kornia, scikit-learn, pandas, pydantic v2 (nouvelle dépendance), PyYAML, pytest, ruff.

**Spec :** `docs/superpowers/specs/2026-09-02-refonte-pipeline-entrainement-design.md`

## Global Constraints

- **Environnement Python** : venv parent partagé `/Users/romain/Projects/Liora/Projet/.venv`. Lancer les tests avec `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`. Installer un paquet avec `VIRTUAL_ENV=/Users/romain/Projects/Liora/Projet/.venv uv pip install <pkg>` (ne **jamais** faire `uv add` / `uv sync` depuis le dossier projet — crée un `.venv` imbriqué sans torch).
- **Branche** : `romain_v2` (déjà créée et courante).
- **`src/config.py` ne doit importer ni `torch`, ni `torchvision`, ni aucun module `src.models` / `src.features`** — data + validation pure uniquement.
- **pydantic v2** : tous les modèles portent `model_config = ConfigDict(extra="forbid")`.
- **Style** : ruff (`line-length = 100`, docstrings google, `from __future__ import annotations` là où utile). Docstrings en français, comme le reste du code.
- **Commits fréquents** : un commit par tâche minimum, message en français préfixé `feat:` / `refactor:` / `test:` / `chore:`.
- **TDD** : test qui échoue → implémentation minimale → test qui passe → commit.
- **Ne pas toucher** : `src/streamlit/`, `src/visualization/`, `notebooks/`, `src/models/predict_model.py`, `src/features/build_features.py`, `src/models/helpers.py` (réutilisé tel quel).
- **Fin de chaque tâche** : `ruff check src tests` et `pytest tests -q` doivent passer.

---

## File Structure

**Créés :**
- `src/config.py` — arbre pydantic `ExperimentConfig` + `load_config` + parsing des overrides `--set`.
- `src/tasks.py` — `TaskSpec` (dataclass frozen) + registre `TASKS` + `get_task`.
- `src/features/dataframe.py` — `create_dataframe`, `make_update_data` (extraits de `dataset.py`).
- `src/features/stats.py` — `compute_dataset_mean_std`, `average_reinhard_target_stats`, `build_reinhard_normalizer`, `build_reinhard`.
- `src/models/__init__.py` (déjà présent, vide) — `build_model(model_cfg, task)`.
- `src/models/baseline.py` — `BaselineClassifier(nn.Module)` (renommé depuis `train_model.py`).
- `src/training/__init__.py` — vide.
- `src/training/trainer.py` — `Trainer`.
- `src/training/callbacks.py` — `EarlyStopping` (déplacé depuis `models/callbacks.py`).
- `src/training/components.py` — `resolve_device`, `split_off_test`, `make_splits`, `TransformBuilder`, `build_optimizer`, `build_scheduler`.
- `src/experiments/__init__.py` — vide.
- `src/experiments/artifacts.py` — `RunResult` (dataclass), `dirname_for`, `write_run_dir`, `append_leaderboard`, `flatten`.
- `src/experiments/runner.py` — `run_experiment`, `_assemble_result`, `_warn_if_missing_classes`, `seed_everything`.
- `src/experiments/grid.py` — `expand_grid`, `run_grid`, `_apply_overrides`.
- `src/experiments/run.py` — CLI run unique (`python -m src.experiments.run`).
- `src/experiments/grid_cli.py` — CLI grid (`python -m src.experiments.grid`).
- `configs/base.yaml`, `configs/grids/preprocessing.yaml`, `configs/grids/augmentation.yaml`, `configs/grids/imbalance.yaml`.
- Tests : `tests/test_config.py`, `tests/test_tasks.py`, `tests/features/test_dataframe.py`, `tests/features/test_dataset.py`, `tests/features/test_stats.py`, `tests/features/test_transform.py`, `tests/models/test_build.py`, `tests/models/test_losses.py`, `tests/models/test_metrics.py`, `tests/training/test_callbacks.py`, `tests/training/test_components.py`, `tests/training/test_trainer.py`, `tests/experiments/test_artifacts.py`, `tests/experiments/test_grid.py`, `tests/experiments/test_runner_smoke.py`.
- `tests/features/__init__.py`, `tests/training/__init__.py`, `tests/experiments/__init__.py`.

**Modifiés :**
- `src/features/dataset.py` — réduit à `BloodCellDataset` (une classe, `task`-driven) + `DatasetTransform`.
- `src/features/transform.py` — `build_transform(cfg: TransformConfig, *, training, ...)` ; helpers Reinhard déplacés vers `stats.py`.
- `src/models/dataset_model.py` — `DatasetClassifier(nn.Module)` (n'hérite plus de `BaseModel`).
- `src/models/losses.py` — ajout `build_loss(loss_cfg, task, train_df)`.
- `src/models/metrics.py` — `build_metrics(task, for_test=False)` remplace `build_metrics(num_classes, device)`.
- `pyproject.toml` — `pydantic>=2` dans `dependencies`.
- `requirements.txt` — `pydantic>=2`.

**Supprimés :**
- `src/train.py`
- `src/models/base_model.py`
- `src/models/train_model.py` (contenu déplacé vers `baseline.py`)
- `src/models/callbacks.py` (déplacé vers `training/callbacks.py`)
- `tests/models/test_train_loop.py` (réécrit en `tests/training/test_trainer.py`)
- `tests/models/test_callbacks.py` (déplacé en `tests/training/test_callbacks.py`)

---

## Task 1: Dépendance pydantic + `src/config.py`

**Files:**
- Modify: `pyproject.toml:6-7` (dependencies), `requirements.txt`
- Create: `src/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: rien.
- Produces:
  - `DataConfig`, `TransformConfig`, `LossConfig`, `OptimConfig`, `SchedulerConfig`, `ModelConfig`, `TrainConfig`, `OutputConfig`, `ExperimentConfig` — pydantic `BaseModel`, champs exactement comme dans la spec §3.
  - `load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig`
  - `parse_overrides(items: list[str]) -> dict` — `["a.b=1e-3"]` → `{"a": {"b": 0.001}}`
  - `deep_merge(base: dict, patch: dict) -> dict` — fusion récursive, `patch` prioritaire, ne mute pas `base`.

- [ ] **Step 1: Installer pydantic**

Run: `VIRTUAL_ENV=/Users/romain/Projects/Liora/Projet/.venv uv pip install 'pydantic>=2'`
Expected: installation OK. Vérifier : `/Users/romain/Projects/Liora/Projet/.venv/bin/python -c "import pydantic; print(pydantic.VERSION)"` → `2.x`.

- [ ] **Step 2: Déclarer la dépendance**

Dans `pyproject.toml`, remplacer `dependencies = []` par :
```toml
dependencies = ["pydantic>=2"]
```
Ajouter une ligne `pydantic>=2` à `requirements.txt`.

- [ ] **Step 3: Écrire les tests qui échouent**

`tests/test_config.py` :
```python
"""Tests du chargement et de la validation de configuration."""

import pytest
from pydantic import ValidationError

from src.config import ExperimentConfig, deep_merge, load_config, parse_overrides


def test_defaults_are_valid():
    cfg = ExperimentConfig()
    assert cfg.task == "type"
    assert cfg.train.nb_epoch == 3
    assert cfg.loss.name == "cross_entropy"
    assert cfg.transform.reinhard is True


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"trian": {"nb_epoch": 5}})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"train": {"nb_epochs": 5}})


def test_parse_overrides_literals():
    out = parse_overrides(["train.lr=1e-3", "loss.name=focal", "transform.reinhard=false"])
    assert out == {
        "train": {"lr": 0.001},
        "loss": {"name": "focal"},
        "transform": {"reinhard": False},
    }


def test_parse_overrides_keeps_bare_string():
    assert parse_overrides(["model.name=baseline"]) == {"model": {"name": "baseline"}}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1, "y": 2}}
    patch = {"a": {"y": 9}, "b": 3}
    merged = deep_merge(base, patch)
    assert merged == {"a": {"x": 1, "y": 9}, "b": 3}
    assert base == {"a": {"x": 1, "y": 2}}


def test_validation_none_with_plateau_is_rejected():
    with pytest.raises(ValidationError, match="validation"):
        ExperimentConfig.model_validate(
            {"train": {"validation": "none"}, "scheduler": {"name": "plateau"}}
        )


def test_validation_none_with_early_stopping_is_rejected():
    with pytest.raises(ValidationError, match="validation"):
        ExperimentConfig.model_validate(
            {"train": {"validation": "none", "early_stopping_patience": 3}}
        )


def test_load_config_applies_overrides(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("task: criticality\ntrain:\n  nb_epoch: 2\n")
    cfg = load_config(cfg_file, ["train.nb_epoch=7", "loss.name=focal"])
    assert cfg.task == "criticality"
    assert cfg.train.nb_epoch == 7
    assert cfg.loss.name == "focal"
```

- [ ] **Step 4: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/test_config.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 5: Implémenter `src/config.py`**

```python
"""Configuration d'expérience : arbre pydantic + chargement YAML + overrides CLI.

Ce module ne doit importer NI torch, NI torchvision, NI aucun module src.models /
src.features — uniquement des données et de la validation.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    datasets: list[Literal["barcelona", "munich"]] = ["barcelona", "munich"]
    n_images: int = 0
    test_size: float = 0.15
    stratify: Literal[
        "no", "type", "dataset", "type_dataset", "criticality", "binary_criticality"
    ] = "type_dataset"


class TransformConfig(_Base):
    img_size: tuple[int, int] = (224, 224)
    reinhard: bool = True
    reinhard_samples: int = 500
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    jpeg: bool = True
    jpeg_quality: int = 35
    p_rotation: float = 0.9
    p_brightness: float = 0.5
    p_contrast: float = 0.5
    brightness_delta: float = 0.15
    contrast_delta: float = 0.15


class LossConfig(_Base):
    name: Literal["cross_entropy", "focal", "cost_sensitive"] = "cross_entropy"
    beta: float = Field(0.9999, gt=0, lt=1)
    gamma: float = 2.0
    extra_weight: dict[str, float] = {}
    lambda_cost: float = 1.0


class OptimConfig(_Base):
    name: Literal["adam", "sgd"] = "adam"
    lr: float = 1e-2
    weight_decay: float = 0.0


class SchedulerConfig(_Base):
    name: Literal["none", "plateau", "cosine", "step"] = "none"
    patience: int = 3
    lr_min: float = 0.0
    step_gamma: float = 0.1


class ModelConfig(_Base):
    name: Literal["baseline", "transfer"] = "baseline"


class TrainConfig(_Base):
    nb_epoch: int = 3
    batch_size: int = 32
    validation: Literal["none", "holdout", "cross"] = "holdout"
    val_size: float = 0.15
    cross_val_folds: int = 3
    early_stopping_patience: int = 0
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class OutputConfig(_Base):
    root: Path = Path("results")
    save_checkpoint: bool = True
    save_figures: bool = True


class ExperimentConfig(_Base):
    task: Literal["dataset", "type", "criticality", "binary_criticality"] = "type"
    seed: int = 0
    data: DataConfig = DataConfig()
    transform: TransformConfig = TransformConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    optim: OptimConfig = OptimConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    train: TrainConfig = TrainConfig()
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def _check_validation_requirements(self) -> ExperimentConfig:
        needs_val = self.train.early_stopping_patience > 0 or self.scheduler.name == "plateau"
        if needs_val and self.train.validation == "none":
            raise ValueError(
                "scheduler='plateau' et early_stopping_patience>0 exigent "
                "train.validation='holdout' ou 'cross'."
            )
        return self


def _coerce(value: str) -> Any:
    """Convertit une valeur d'override en littéral Python, sinon la laisse en str."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_overrides(items: list[str]) -> dict:
    """``["a.b=1e-3", "c=focal"]`` -> ``{"a": {"b": 0.001}, "c": "focal"}``."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override invalide (attendu clé=valeur) : {item!r}")
        dotted, raw = item.split("=", 1)
        node = out
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _coerce(raw)
    return out


def deep_merge(base: dict, patch: dict) -> dict:
    """Fusion récursive ; ``patch`` prioritaire ; ``base`` non muté."""
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Charge un YAML, applique les overrides ``clé.pointée=valeur``, valide."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    merged = deep_merge(raw, parse_overrides(overrides or []))
    return ExperimentConfig.model_validate(merged)
```

- [ ] **Step 6: Vérifier que les tests passent**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/test_config.py -q`
Expected: PASS (8 tests)

- [ ] **Step 7: Lint + suite complète**

Run: `ruff check src/config.py tests/test_config.py && /Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`
Expected: PASS (23 tests)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml requirements.txt src/config.py tests/test_config.py uv.lock
git commit -m "feat: config d'experience pydantic + chargement YAML/overrides"
```

---

## Task 2: `src/tasks.py`

**Files:**
- Create: `src/tasks.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Consumes: `src.constants` (`FINAL_CLASSES`, `CRITICALITY_CLASSES`, `BINARY_CRITICALITY_CLASSES`, `BARCELONA_KEY`, `MUNICH_KEY`).
- Produces:
  - `TaskSpec` — `@dataclass(frozen=True)` : `name: str`, `label_column: str`, `class_names: list[str]`, `num_classes: int`, `kind: Literal["binary","multiclass"]`, `label_dtype: torch.dtype`, `target_shape: Literal["scalar","vector1"]`, `default_loss: Literal["cross_entropy","bce"]`.
  - `TASKS: dict[str, TaskSpec]` — clés `"dataset"`, `"type"`, `"criticality"`, `"binary_criticality"`.
  - `get_task(name: str) -> TaskSpec` — lève `KeyError` avec message explicite si inconnu.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_tasks.py` :
```python
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
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/test_tasks.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.tasks'`

- [ ] **Step 3: Implémenter `src/tasks.py`**

```python
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
```

- [ ] **Step 4: Vérifier que les tests passent**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/test_tasks.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint + commit**

```bash
ruff check src/tasks.py tests/test_tasks.py
git add src/tasks.py tests/test_tasks.py
git commit -m "feat: TaskSpec + registre des taches"
```

---

## Task 3: `src/features/dataframe.py`

**Files:**
- Create: `src/features/dataframe.py`, `tests/features/__init__.py`, `tests/features/test_dataframe.py`
- Modify: `src/features/dataset.py` (retirer `create_dataframe`, `make_update_data`, `get_images` ; garder un ré-export temporaire)

**Interfaces:**
- Consumes: `src.config.DataConfig`, `src.tasks.TaskSpec`, `src.constants`.
- Produces:
  - `create_dataframe(data_cfg: DataConfig, task: TaskSpec) -> pd.DataFrame` — colonnes `path, dataset, original_type, type, label, binary_criticality, criticality, type_dataset` ; `dropna` sur les lignes non mappées ; sous-échantillonnage si `data_cfg.n_images > 0` (stratifié sur `data_cfg.stratify` si `!= "no"`).
  - `make_update_data(task: TaskSpec) -> Callable[[pd.Series], tuple]`
  - `get_images(data_dir, dataset_name) -> list[tuple[str, str, str]]`

- [ ] **Step 1: Créer `tests/features/__init__.py` (vide) et écrire le test qui échoue**

`tests/features/test_dataframe.py` :
```python
"""Tests de construction du DataFrame d'images."""

import pytest

from src.config import DataConfig
from src.constants import DATASET_PATHS
from src.tasks import get_task

pytestmark = pytest.mark.skipif(
    not DATASET_PATHS["barcelona"].exists(), reason="données absentes"
)


def test_create_dataframe_columns_and_labels():
    from src.features.dataframe import create_dataframe

    df = create_dataframe(DataConfig(n_images=200, stratify="type"), get_task("type"))
    assert not df.empty
    assert {"path", "dataset", "label", "type", "criticality"} <= set(df.columns)
    assert df["label"].between(0, 9).all()
    assert df["label"].notna().all()


def test_create_dataframe_task_criticality_labels_in_range():
    from src.features.dataframe import create_dataframe

    df = create_dataframe(DataConfig(n_images=200, stratify="no"), get_task("criticality"))
    assert df["label"].between(0, 2).all()


def test_n_images_zero_returns_full_dataset():
    from src.features.dataframe import create_dataframe

    df = create_dataframe(DataConfig(n_images=0), get_task("type"))
    assert len(df) > 1000
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/features/test_dataframe.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.features.dataframe'`

- [ ] **Step 3: Créer `src/features/dataframe.py`**

Déplacer depuis `src/features/dataset.py` : `make_update_data`, `get_images`, `create_dataframe`. Adapter les signatures :

```python
"""Construction du DataFrame d'images (chemins + labels dérivés)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd

from src.constants import (
    BARCELONA_KEY,
    BINARY_CRITICALITY,
    CRITICALITY,
    DATASET_PATHS,
    MAPPING_DICT,
    MODE_TO_LABEL,
    MUNICH_KEY,
)

if TYPE_CHECKING:
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
```

Note : `data_cfg.datasets` remplace la logique `use_dataset in ["all", ...]`. `"all"` = liste par défaut `["barcelona", "munich"]`.

- [ ] **Step 4: Retirer les fonctions déplacées de `dataset.py`, ré-export temporaire**

Dans `src/features/dataset.py`, supprimer `make_update_data`, `get_images`, `create_dataframe`, et ajouter en tête (pour ne pas casser `train.py` encore présent) :
```python
from src.features.dataframe import create_dataframe, get_images, make_update_data  # noqa: F401  (compat, retiré en Task 12)
```

- [ ] **Step 5: Vérifier tests + non-régression**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`
Expected: PASS (les 3 nouveaux tests passent si les données sont présentes, sinon `skip` ; les 29 précédents inchangés)

- [ ] **Step 6: Lint + commit**

```bash
ruff check src/features/dataframe.py src/features/dataset.py tests/features
git add src/features/dataframe.py src/features/dataset.py tests/features/__init__.py tests/features/test_dataframe.py
git commit -m "refactor: extraire create_dataframe dans features/dataframe.py"
```

---

## Task 4: `BloodCellDataset` piloté par `TaskSpec`

**Files:**
- Modify: `src/features/dataset.py`
- Test: `tests/features/test_dataset.py`

**Interfaces:**
- Consumes: `src.tasks.TaskSpec`.
- Produces:
  - `BloodCellDataset(data: pd.DataFrame, task: TaskSpec, transform=None)` — `__getitem__` renvoie `(image_tensor, label_tensor)` ; `label` de dtype `task.label_dtype` ; forme `(1,)` si `task.target_shape == "vector1"`, scalaire sinon. Méthodes conservées : `get_details(idx)`, `get_num_classes()`. Attribut `.data`.
  - `DatasetTransform(subset, transform=None)` — inchangé.
- Supprimés : `DomainDataset`, `CriticalityDataset`.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/features/test_dataset.py` :
```python
"""Tests du Dataset image piloté par TaskSpec."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.features.dataset import BloodCellDataset, DatasetTransform
from src.tasks import get_task


@pytest.fixture
def tiny_df(tmp_path):
    rows = []
    for i in range(4):
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(np.uint8(np.random.rand(20, 20, 3) * 255)).save(p)
        rows.append({"path": str(p), "label": i % 2})
    return pd.DataFrame(rows)


def test_label_dtype_scalar_for_multiclass(tiny_df):
    ds = BloodCellDataset(tiny_df, get_task("type"))
    _, label = ds[0]
    assert label.dtype == torch.long
    assert label.ndim == 0


def test_label_shape_vector1_for_dataset_task(tiny_df):
    ds = BloodCellDataset(tiny_df, get_task("dataset"))
    _, label = ds[0]
    assert label.dtype == torch.float32
    assert label.shape == (1,)


def test_transform_is_applied(tiny_df):
    ds = BloodCellDataset(tiny_df, get_task("type"), transform=lambda x: torch.zeros(3, 8, 8))
    image, _ = ds[0]
    assert image.shape == (3, 8, 8)


def test_dataset_transform_wraps_subset(tiny_df):
    base = BloodCellDataset(tiny_df, get_task("type"))
    wrapped = DatasetTransform(base, transform=lambda x: x * 0)
    image, _ = wrapped[0]
    assert torch.equal(image, torch.zeros_like(image))
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/features/test_dataset.py -q`
Expected: FAIL (`BloodCellDataset` prend `transform=` en 2e position aujourd'hui, pas `task`)

- [ ] **Step 3: Réécrire `src/features/dataset.py`**

```python
"""Dataset image pour la classification de cellules sanguines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from src.features.dataframe import create_dataframe, get_images, make_update_data  # noqa: F401

if TYPE_CHECKING:
    import pandas as pd

    from src.tasks import TaskSpec


class BloodCellDataset(Dataset):
    """Renvoie ``(image, label)`` ; type et forme du label dictés par ``task``."""

    def __init__(self, data: pd.DataFrame, task: TaskSpec, transform=None):
        self.data = data
        self.task = task
        self.transforms = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        with Image.open(row.path) as img:
            image = v2.functional.pil_to_tensor(img.convert("RGB"))
        if self.transforms:
            image = self.transforms(image)
        label = torch.tensor(row[self.task.label_column], dtype=self.task.label_dtype)
        if self.task.target_shape == "vector1":
            label = label.unsqueeze(0)
        return image, label

    def get_details(self, idx: int):
        """Détails de l'image à l'index ``idx``."""
        return self.data.iloc[idx]

    def get_num_classes(self) -> int:
        """Nombre de classes distinctes présentes dans le DataFrame."""
        return self.data[self.task.label_column].nunique()


class DatasetTransform(Dataset):
    """Applique un transform à la volée sur un ``Subset`` / ``Dataset``."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
```

Note : `compute_dataset_mean_std` part en Task 5 ; s'il est encore ici au début de la tâche, le laisser jusqu'à ce que Task 5 le déplace (ou le déplacer maintenant si l'ordre le permet — Task 5 suit).

- [ ] **Step 4: Vérifier les tests**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/features/test_dataset.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Suite complète**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`
Expected: `src/train.py` importe encore `BloodCellDataset(train_data)` sans `task` → il n'est pas testé directement, donc la suite passe. Si un test casse, noter et corriger l'appelant.

- [ ] **Step 6: Lint + commit**

```bash
ruff check src/features/dataset.py tests/features/test_dataset.py
git add src/features/dataset.py tests/features/test_dataset.py
git commit -m "refactor: BloodCellDataset unique pilote par TaskSpec"
```

---

## Task 5: `src/features/stats.py`

**Files:**
- Create: `src/features/stats.py`, `tests/features/test_stats.py`
- Modify: `src/features/dataset.py` (retirer `compute_dataset_mean_std`), `src/features/transform.py` (retirer `average_reinhard_target_stats`, `build_reinhard_normalizer`)

**Interfaces:**
- Consumes: `src.config.TransformConfig`, `src.tasks.TaskSpec`, `src.features.dataset.BloodCellDataset`.
- Produces:
  - `compute_dataset_mean_std(train_df, transform_cfg, task, reinhard_normalizer=None, n_batches=10, batch_size=64) -> tuple[torch.Tensor, torch.Tensor]`
  - `average_reinhard_target_stats(normalizer, target_image_paths, img_size=(224,224), n_samples=100, origins=None, seed=0)` — inchangé (déplacé).
  - `build_reinhard_normalizer(target_image_path, img_size=(224,224), method=None)` — inchangé (déplacé).
  - `build_reinhard(train_df, transform_cfg) -> ReinhardNormalizer` — instancie `ReinhardNormalizer(backend="torch", method=None)` puis `average_reinhard_target_stats` sur `train_df.path` avec `origins=train_df.dataset`, `n_samples=transform_cfg.reinhard_samples`.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/features/test_stats.py` :
```python
"""Tests des statistiques de normalisation."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.config import TransformConfig
from src.tasks import get_task


@pytest.fixture
def tiny_df(tmp_path):
    rows = []
    for i in range(8):
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(np.uint8(np.random.rand(30, 30, 3) * 255)).save(p)
        rows.append({"path": str(p), "label": i % 2, "dataset": "barcelona"})
    return pd.DataFrame(rows)


def test_compute_dataset_mean_std_finite(tiny_df):
    from src.features.stats import compute_dataset_mean_std

    cfg = TransformConfig(reinhard=False, clahe=False, jpeg=False)
    mean, std = compute_dataset_mean_std(tiny_df, cfg, get_task("type"), n_batches=2, batch_size=4)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert (std >= 0).all()
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/features/test_stats.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.features.stats'`

- [ ] **Step 3: Créer `src/features/stats.py`**

Déplacer `compute_dataset_mean_std` (depuis `dataset.py`), `average_reinhard_target_stats` et `build_reinhard_normalizer` (depuis `transform.py`). Adapter `compute_dataset_mean_std` :

```python
"""Statistiques calées sur le train : mean/std du dataset, cible Reinhard."""

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING

import torch
import torchstain
from torch.utils.data import DataLoader

from src.constants import IMG_SIZE
from src.features.dataset import BloodCellDataset
from src.features.transform import build_transform, load_image_rgb, resize_with_pad

if TYPE_CHECKING:
    import pandas as pd

    from src.config import TransformConfig
    from src.tasks import TaskSpec


def compute_dataset_mean_std(
    train_df: pd.DataFrame,
    transform_cfg: TransformConfig,
    task: TaskSpec,
    reinhard_normalizer=None,
    n_batches: int = 10,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Moyenne / écart-type par canal, calculés en neutralisant la normalisation finale."""
    probe_transform = build_transform(
        transform_cfg,
        training=False,
        reinhard_normalizer=reinhard_normalizer,
        dataset_mean=torch.zeros(3),
        dataset_std=torch.ones(3),
    )
    probe_ds = BloodCellDataset(train_df, task, transform=probe_transform)
    loader = DataLoader(probe_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    for images, _ in itertools.islice(loader, n_batches):
        images = images.double()
        n_pixels += images.shape[0] * images.shape[2] * images.shape[3]
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images**2).sum(dim=(0, 2, 3))

    mean = channel_sum / n_pixels
    std = torch.sqrt((channel_sq_sum / n_pixels - mean**2).clamp(min=0))
    return mean.float(), std.float()


def average_reinhard_target_stats(
    normalizer, target_image_paths, img_size=IMG_SIZE, n_samples=100, origins=None, seed=0
):
    """Moyenne les stats Lab cible de Reinhard sur un échantillon (équilibré par origine)."""
    rng = random.Random(seed)
    if origins is not None:
        by_origin: dict = {}
        for path, origin in zip(target_image_paths, origins):
            by_origin.setdefault(origin, []).append(path)
        n_per_origin = max(1, n_samples // len(by_origin))
        sample_paths = []
        for origin_paths in by_origin.values():
            sample_paths += rng.sample(origin_paths, min(n_per_origin, len(origin_paths)))
    else:
        paths = list(target_image_paths)
        sample_paths = rng.sample(paths, min(n_samples, len(paths)))

    means, stds = [], []
    for path in sample_paths:
        img = resize_with_pad(load_image_rgb(path), img_size, fill=255).float()
        normalizer.fit(img)
        means.append(normalizer.target_means)
        stds.append(normalizer.target_stds)
    normalizer.target_means = torch.stack(means).mean(dim=0)
    normalizer.target_stds = torch.stack(stds).mean(dim=0)
    return normalizer


def build_reinhard_normalizer(target_image_path, img_size=IMG_SIZE, method=None):
    """Fitte un ``ReinhardNormalizer`` sur une image de référence unique."""
    target = resize_with_pad(load_image_rgb(target_image_path), img_size, fill=255).float()
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend="torch", method=method)
    normalizer.fit(target)
    return normalizer


def build_reinhard(train_df: pd.DataFrame, transform_cfg: TransformConfig):
    """Cible Reinhard moyennée sur le train, équilibrée entre datasets d'origine."""
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend="torch", method=None)
    return average_reinhard_target_stats(
        normalizer,
        list(train_df.path),
        origins=list(train_df.dataset),
        n_samples=transform_cfg.reinhard_samples,
    )
```

- [ ] **Step 4: Nettoyer `dataset.py` et `transform.py`**

- `src/features/dataset.py` : supprimer `compute_dataset_mean_std` et l'import `DataLoader`, `itertools`, `time` s'ils ne servent plus.
- `src/features/transform.py` : supprimer `average_reinhard_target_stats` et `build_reinhard_normalizer` (gardés seulement `apply_reinhard`, les classes `*Transform`, `build_transform`, `load_image_rgb`, `resize_with_pad`). Supprimer l'import `random` si inutilisé.
- Mettre à jour `src/train.py` : `from src.features.stats import average_reinhard_target_stats, compute_dataset_mean_std` (au lieu des anciens emplacements) — il reste fonctionnel jusqu'à sa suppression en Task 17.

- [ ] **Step 5: Vérifier tests + non-régression**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Lint + commit**

```bash
ruff check src/features tests/features/test_stats.py src/train.py
git add src/features/stats.py src/features/dataset.py src/features/transform.py src/train.py tests/features/test_stats.py
git commit -m "refactor: regrouper les stats de normalisation dans features/stats.py"
```

---

## Task 6: `build_transform(cfg: TransformConfig, ...)`

**Files:**
- Modify: `src/features/transform.py`, `src/features/stats.py` (appel), `src/train.py` (appels)
- Test: `tests/features/test_transform.py`

**Interfaces:**
- Consumes: `src.config.TransformConfig`.
- Produces:
  - `build_transform(cfg: TransformConfig, *, training: bool, reinhard_normalizer=None, dataset_mean=None, dataset_std=None) -> v2.Compose`
    - `dataset_mean` / `dataset_std` par défaut `torch.tensor([0.5,0.5,0.5])` / `torch.tensor([0.2,0.2,0.2])` (placeholders actuels).
    - Ordre inchangé : ResizeWithPad → (JPEG si `cfg.jpeg`) → ToUnitRange → (si `training` : RandomApply(RandomRotation, p=`cfg.p_rotation`), HFlip, VFlip) → (ReinhardTransform si `reinhard_normalizer`) → (CLAHETransform si `cfg.clahe`) → (si `training` : RandomApply(ColorJitter(brightness=`cfg.brightness_delta`), p=`cfg.p_brightness`), RandomApply(ColorJitter(contrast=`cfg.contrast_delta`), p=`cfg.p_contrast`)) → Normalize.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/features/test_transform.py` :
```python
"""Tests de la fabrique de transforms."""

import torch
from torchvision.transforms import v2

from src.config import TransformConfig
from src.features.transform import build_transform


def test_train_transform_has_augmentation_steps():
    cfg = TransformConfig(reinhard=False, clahe=False)
    t = build_transform(cfg, training=True)
    kinds = [type(s).__name__ for s in t.transforms]
    assert kinds.count("RandomApply") >= 1
    assert "RandomHorizontalFlip" in kinds


def test_eval_transform_has_no_random_flip():
    cfg = TransformConfig(reinhard=False, clahe=False)
    t = build_transform(cfg, training=False)
    kinds = [type(s).__name__ for s in t.transforms]
    assert "RandomHorizontalFlip" not in kinds
    assert kinds[-1] == "Normalize"


def test_clahe_toggle():
    with_clahe = build_transform(TransformConfig(reinhard=False, clahe=True), training=False)
    without = build_transform(TransformConfig(reinhard=False, clahe=False), training=False)
    names = lambda c: [type(s).__name__ for s in c.transforms]
    assert "CLAHETransform" in names(with_clahe)
    assert "CLAHETransform" not in names(without)


def test_transform_runs_on_a_tensor():
    cfg = TransformConfig(reinhard=False, clahe=False, jpeg=False)
    t = build_transform(cfg, training=True)
    out = t(torch.randint(0, 255, (3, 40, 40), dtype=torch.uint8))
    assert out.shape == (3, 224, 224)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/features/test_transform.py -q`
Expected: FAIL (`build_transform` prend `training` en 1er argument positionnel, pas `cfg`)

- [ ] **Step 3: Adapter `build_transform`**

Remplacer la signature et le corps de `build_transform` dans `src/features/transform.py` :

```python
def build_transform(
    cfg: TransformConfig,
    *,
    training: bool,
    reinhard_normalizer=None,
    dataset_mean=None,
    dataset_std=None,
) -> v2.Compose:
    """Construit le ``v2.Compose`` complet (resize -> aug -> Reinhard/CLAHE -> normalize).

    Seule la partie augmentation (rotation / flip / jitter) est conditionnée par
    ``training`` ; Reinhard / CLAHE / normalisation sont identiques sur tous les splits.
    """
    if dataset_mean is None:
        dataset_mean = torch.tensor([0.5, 0.5, 0.5])
    if dataset_std is None:
        dataset_std = torch.tensor([0.2, 0.2, 0.2])

    steps: list = [ResizeWithPad(cfg.img_size, fill=255)]
    if cfg.jpeg:
        steps.append(v2.JPEG(cfg.jpeg_quality))
    steps.append(ToUnitRange())

    if training:
        steps += [
            v2.RandomApply(
                [v2.RandomRotation(degrees=180, fill=1.0, interpolation=InterpolationMode.BILINEAR)],
                p=cfg.p_rotation,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ]

    if reinhard_normalizer is not None:
        steps.append(ReinhardTransform(reinhard_normalizer))
    if cfg.clahe:
        steps.append(CLAHETransform(cfg.clahe_clip_limit, cfg.clahe_grid_size))

    if training:
        steps += [
            v2.RandomApply([v2.ColorJitter(brightness=cfg.brightness_delta)], p=cfg.p_brightness),
            v2.RandomApply([v2.ColorJitter(contrast=cfg.contrast_delta)], p=cfg.p_contrast),
        ]

    mean = dataset_mean.tolist() if torch.is_tensor(dataset_mean) else list(dataset_mean)
    std = dataset_std.tolist() if torch.is_tensor(dataset_std) else list(dataset_std)
    steps.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(steps)
```

Ajouter `from typing import TYPE_CHECKING` + bloc `if TYPE_CHECKING: from src.config import TransformConfig`. Retirer les constantes `DATASET_MEAN` / `DATASET_STD` inutilisées (ou les garder si `stats.py` les importe — il ne les importe pas).

- [ ] **Step 4: Corriger les appelants**

- `src/features/stats.py` : l'appel dans `compute_dataset_mean_std` est déjà à la nouvelle signature (écrit en Task 5).
- `src/train.py` : dans `transform_fn`, remplacer `build_transform(training=training, reinhard_normalizer=normalizer, dataset_mean=..., dataset_std=...)` par `build_transform(TransformConfig(reinhard=use_reinhard, clahe=use_clahe), training=training, reinhard_normalizer=normalizer, dataset_mean=dataset_mean, dataset_std=dataset_std)`. Import `from src.config import TransformConfig`. (Rustine jusqu'à Task 17.)

- [ ] **Step 5: Vérifier tests + non-régression**

Run: `ruff check src/features && /Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`
Expected: PASS (4 nouveaux tests + le reste)

- [ ] **Step 6: Commit**

```bash
git add src/features/transform.py src/features/stats.py src/train.py tests/features/test_transform.py
git commit -m "refactor: build_transform pilote par TransformConfig"
```

---

## Task 7: Modèles purs + `build_model`

**Files:**
- Create: `src/models/baseline.py`
- Modify: `src/models/__init__.py`, `src/models/dataset_model.py`
- Delete: `src/models/train_model.py`
- Test: `tests/models/test_build.py`

**Interfaces:**
- Consumes: `src.config.ModelConfig`, `src.tasks.TaskSpec`.
- Produces:
  - `src.models.baseline.BaselineClassifier(num_classes: int)` — `nn.Module` pur (le corps `nn.Sequential` actuel, `num_class` → `num_classes`).
  - `src.models.dataset_model.DatasetClassifier(img_size=(224,224), channels=3, nb_conv=2)` — `nn.Module` pur (retirer `class BaseModel` de l'héritage, garder le corps).
  - `src.models.build_model(model_cfg: ModelConfig, task: TaskSpec) -> nn.Module` :
    - `model_cfg.name == "dataset"` → `DatasetClassifier()`
    - `model_cfg.name == "baseline"` → `BaselineClassifier(task.num_classes)`
    - `model_cfg.name == "transfer"` → `raise NotImplementedError("model 'transfer' : à implémenter")`

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/models/test_build.py` :
```python
"""Tests de la fabrique de modèles."""

import pytest
import torch
from torch import nn

from src.config import ModelConfig
from src.models import build_model
from src.tasks import get_task


def test_baseline_output_dim_matches_task():
    model = build_model(ModelConfig(name="baseline"), get_task("type"))
    assert isinstance(model, nn.Module)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)


def test_dataset_task_builds_single_logit_model():
    model = build_model(ModelConfig(name="baseline"), get_task("dataset"))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_transfer_not_implemented():
    with pytest.raises(NotImplementedError, match="transfer"):
        build_model(ModelConfig(name="transfer"), get_task("type"))


def test_models_are_plain_nn_module():
    from src.models.baseline import BaselineClassifier

    assert BaselineClassifier.__mro__[1] is nn.Module
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/models/test_build.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.models.baseline'`

- [ ] **Step 3: Créer `src/models/baseline.py`**

```python
"""Modèle baseline (CNN maison) pour comparer le préprocessing."""

from __future__ import annotations

from torch import nn


class BaselineClassifier(nn.Module):
    """Petit CNN de référence."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 27 * 27, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x)
```

- [ ] **Step 4: `dataset_model.py` en `nn.Module` pur**

Dans `src/models/dataset_model.py` : remplacer `from src.models.base_model import BaseModel` par `from torch import nn` (déjà présent), et `class DatasetClassifier(BaseModel)` par `class DatasetClassifier(nn.Module)`. Le reste inchangé. Supprimer le gros bloc commenté en fin de fichier.

- [ ] **Step 5: `src/models/__init__.py`**

```python
"""Fabrique de modèles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.baseline import BaselineClassifier
from src.models.dataset_model import DatasetClassifier

if TYPE_CHECKING:
    from torch import nn

    from src.config import ModelConfig
    from src.tasks import TaskSpec


def build_model(model_cfg: ModelConfig, task: TaskSpec) -> nn.Module:
    """Instancie le modèle correspondant à la config et à la tâche."""
    if task.name == "dataset":
        return DatasetClassifier()
    if model_cfg.name == "baseline":
        return BaselineClassifier(task.num_classes)
    if model_cfg.name == "transfer":
        raise NotImplementedError("model 'transfer' : à implémenter")
    raise ValueError(f"modèle inconnu : {model_cfg.name!r}")
```

- [ ] **Step 6: Supprimer `train_model.py`, corriger `train.py`**

```bash
git rm src/models/train_model.py
```
Dans `src/train.py` : `from src.models.train_model import BaselineClassifier` → `from src.models.baseline import BaselineClassifier`. `BaselineClassifier(num_class=...)` → `BaselineClassifier(...)`. `BaselineClassifier` hérite maintenant de `nn.Module` pur : `train.py` appelle `model.fit(...)` — **ce chemin va casser** ; c'est attendu, `train.py` est supprimé en Task 17. Pour garder la suite verte d'ici là, envelopper l'appel : si un test échoue à cause de ça, ce n'est pas le cas (aucun test n'exécute `train.py`). Vérifier à l'étape suivante.

- [ ] **Step 7: Vérifier tests**

Run: `ruff check src/models tests/models/test_build.py && /Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`
Expected: PASS (4 nouveaux). `tests/models/test_train_loop.py` et `test_callbacks.py` importent encore `src.models.base_model` / `src.models.callbacks` — **ne pas y toucher ici**, ils passent encore.

- [ ] **Step 8: Commit**

```bash
git add src/models/baseline.py src/models/__init__.py src/models/dataset_model.py src/train.py tests/models/test_build.py
git commit -m "refactor: modeles en nn.Module purs + build_model"
```

---

## Task 8: `build_loss`

**Files:**
- Modify: `src/models/losses.py`
- Test: `tests/models/test_losses.py`

**Interfaces:**
- Consumes: `src.config.LossConfig`, `src.tasks.TaskSpec`, un `pd.DataFrame` d'entraînement.
- Produces:
  - `build_loss(loss_cfg: LossConfig, task: TaskSpec, train_df: pd.DataFrame) -> nn.Module` :
    - `task.default_loss == "bce"` (tâche `dataset`) → `nn.BCEWithLogitsLoss()`
    - `loss_cfg.name == "cross_entropy"` → `nn.CrossEntropyLoss()`
    - `loss_cfg.name == "focal"` → `ClassBalancedFocalLoss(samples_per_cls=<counts par classe dans l'ordre de task.class_names>, class_names=task.class_names, no_of_classes=task.num_classes, beta=loss_cfg.beta, gamma=loss_cfg.gamma, extra_weight=loss_cfg.extra_weight or None)`
    - `loss_cfg.name == "cost_sensitive"` → `CostSensitiveCriticalityLoss(num_classes=task.num_classes, samples_per_criticality=<counts>, lambda_cost=loss_cfg.lambda_cost, class_names=task.class_names, beta=loss_cfg.beta)`

Note : `ClassBalancedFocalLoss.__init__` actuel ignore `no_of_classes` (calcule depuis `counts`). Passer les `counts` dans l'ordre de `task.class_names`, en comblant à 0 les classes absentes du `train_df`.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/models/test_losses.py` :
```python
"""Tests de la fabrique de loss et des pertes custom."""

import pandas as pd
import pytest
import torch
from torch import nn

from src.config import LossConfig
from src.models.losses import ClassBalancedFocalLoss, build_loss
from src.tasks import get_task


def _df(labels):
    return pd.DataFrame({"label": labels, "type": ["x"] * len(labels)})


def test_cross_entropy_default():
    loss = build_loss(LossConfig(), get_task("type"), _df([0, 1, 2, 3]))
    assert isinstance(loss, nn.CrossEntropyLoss)


def test_dataset_task_uses_bce():
    loss = build_loss(LossConfig(), get_task("dataset"), _df([0, 1, 0, 1]))
    assert isinstance(loss, nn.BCEWithLogitsLoss)


def test_focal_loss_dimension_matches_task():
    task = get_task("type")
    loss = build_loss(LossConfig(name="focal", beta=0.99), task, _df(list(range(10)) * 3))
    assert isinstance(loss, ClassBalancedFocalLoss)
    assert loss.class_weights.shape == (task.num_classes,)
    logits = torch.randn(4, task.num_classes)
    target = torch.randint(0, task.num_classes, (4,))
    assert torch.isfinite(loss(logits, target))


def test_focal_handles_missing_class_in_train_df():
    # seules 3 classes sur 10 présentes -> pas de division par zéro
    task = get_task("type")
    loss = build_loss(LossConfig(name="focal"), task, _df([0, 0, 1, 2, 2, 2]))
    assert loss.class_weights.shape == (10,)
    assert torch.isfinite(loss.class_weights).all()
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/models/test_losses.py -q`
Expected: FAIL `ImportError: cannot import name 'build_loss'`

- [ ] **Step 3: Ajouter `build_loss` à `src/models/losses.py`**

En tête, ajouter les imports :
```python
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    import pandas as pd

    from src.config import LossConfig
    from src.tasks import TaskSpec
```

En fin de fichier (avant le `if __name__ == "__main__"`), ajouter :
```python
def _counts_in_task_order(train_df: pd.DataFrame, task: TaskSpec) -> list[int]:
    """Effectifs par classe, dans l'ordre de ``task.class_names`` (0 si absente)."""
    seen = train_df[task.label_column].value_counts().to_dict()
    return [int(seen.get(i, 0)) for i in range(task.num_classes)]


def build_loss(loss_cfg: LossConfig, task: TaskSpec, train_df: pd.DataFrame) -> nn.Module:
    """Instancie la loss correspondant à la config et à la tâche."""
    if task.default_loss == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_cfg.name == "cross_entropy":
        return nn.CrossEntropyLoss()

    counts = _counts_in_task_order(train_df, task)

    if loss_cfg.name == "focal":
        return ClassBalancedFocalLoss(
            samples_per_class=counts,
            class_names=list(task.class_names),
            beta=loss_cfg.beta,
            gamma=loss_cfg.gamma,
            extra_weight=loss_cfg.extra_weight or None,
        )
    if loss_cfg.name == "cost_sensitive":
        return CostSensitiveCriticalityLoss(
            num_classes=task.num_classes,
            samples_per_criticality={n: c for n, c in zip(task.class_names, counts)},
            lambda_cost=loss_cfg.lambda_cost,
            class_names=list(task.class_names),
            beta=loss_cfg.beta,
        )
    raise ValueError(f"loss inconnue : {loss_cfg.name!r}")
```

Note : `effective_number_weights` fait `1 - beta**count` ; `count=0` → `effective_num=0` → division par zéro. Corriger `effective_number_weights` pour remplacer les effectifs nuls par 1 avant calcul :
```python
counts = np.asarray(samples_per_class, dtype=np.float64)
counts = np.where(counts == 0, 1.0, counts)
```
(insérer juste après `counts = np.asarray(...)`).

- [ ] **Step 4: Vérifier tests**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/models/test_losses.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint + suite + commit**

```bash
ruff check src/models/losses.py tests/models/test_losses.py
/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q
git add src/models/losses.py tests/models/test_losses.py
git commit -m "feat: build_loss selon LossConfig et TaskSpec"
```

---

## Task 9: `build_metrics(task, for_test=False)`

**Files:**
- Modify: `src/models/metrics.py`, `src/train.py` (appels)
- Test: `tests/models/test_metrics.py`

**Interfaces:**
- Consumes: `src.tasks.TaskSpec`.
- Produces:
  - `build_metrics(task: TaskSpec, for_test: bool = False, device="cpu") -> MetricCollection` :
    - `task.kind == "binary"` (tâche `dataset`) → `MetricCollection({"acc": Accuracy(task="binary")})`
    - sinon, `for_test=False` → `MetricCollection({"acc": MulticlassAccuracy(num_classes=task.num_classes), "f1": MulticlassF1Score(num_classes=task.num_classes)})` (métriques légères par epoch)
    - sinon, `for_test=True` → la collection riche actuelle (`f1_macro`, `mcc`, `*_per_class`, `auprc_per_class`, `confusion_matrix`) dimensionnée sur `task.num_classes`.
  - `display_metrics(...)` — inchangé (déjà paramétré par `num_classes`).

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/models/test_metrics.py` :
```python
"""Tests de la fabrique de métriques."""

import torch
from torchmetrics import MetricCollection

from src.models.metrics import build_metrics
from src.tasks import get_task


def test_epoch_metrics_are_lightweight():
    mc = build_metrics(get_task("type"))
    assert isinstance(mc, MetricCollection)
    assert set(mc) == {"acc", "f1"}


def test_test_metrics_are_rich():
    mc = build_metrics(get_task("type"), for_test=True)
    assert "confusion_matrix" in mc
    assert "f1_macro" in mc


def test_binary_task_metrics():
    mc = build_metrics(get_task("dataset"))
    preds = torch.tensor([0.9, 0.1, 0.8])
    target = torch.tensor([1, 0, 1])
    mc.update(preds, target)
    assert "acc" in mc.compute() or "acc" in {k.split("_")[-1] for k in mc.compute()}
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/models/test_metrics.py -q`
Expected: FAIL (`build_metrics` attend `num_classes: int` en 1er argument)

- [ ] **Step 3: Adapter `build_metrics`**

Dans `src/models/metrics.py`, remplacer la fonction `build_metrics` :
```python
from torchmetrics.classification import Accuracy, MulticlassAccuracy, MulticlassF1Score

if TYPE_CHECKING:
    from src.tasks import TaskSpec


def build_metrics(
    task: TaskSpec, for_test: bool = False, device: str | torch.device = "cpu"
) -> MetricCollection:
    """Métriques d'évaluation. ``for_test`` -> collection riche (per-class, confusion)."""
    if task.kind == "binary":
        return MetricCollection({"acc": Accuracy(task="binary")}).to(device)

    n = task.num_classes
    if not for_test:
        return MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=n),
                "f1": MulticlassF1Score(num_classes=n),
            }
        ).to(device)

    metrics = MetricCollection(
        {
            "f1_macro": MulticlassF1Score(num_classes=n, average="macro"),
            "mcc": MulticlassMatthewsCorrCoef(num_classes=n),
            "recall_per_class": MulticlassRecall(num_classes=n, average=None),
            "precision_per_class": MulticlassPrecision(num_classes=n, average=None),
            "fbeta2_per_class": MulticlassFBetaScore(beta=2.0, num_classes=n, average=None),
            "auprc_per_class": MulticlassAveragePrecision(num_classes=n, average=None),
            "confusion_matrix": MulticlassConfusionMatrix(num_classes=n),
        }
    )
    return metrics.to(device)
```

Garder `build_binary_metrics`, `to_binary_abnormal`, `evaluate`, `display_metrics`, etc. inchangés.

- [ ] **Step 4: Corriger `src/train.py`**

`build_metrics(num_classes=..., device=device)` → `build_metrics(get_task(mode), for_test=True, device=device)` pour l'appel test ; l'appel epoch inline (`MetricCollection({...})`) peut rester ou devenir `build_metrics(get_task(mode))`. Import `from src.tasks import get_task`. (Rustine jusqu'à Task 17.)

- [ ] **Step 5: Vérifier tests + non-régression**

Run: `ruff check src/models/metrics.py && /Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/models/metrics.py src/train.py tests/models/test_metrics.py
git commit -m "refactor: build_metrics pilote par TaskSpec (epoch vs test)"
```

---

## Task 10: Déplacer `EarlyStopping` dans `src/training/`

**Files:**
- Create: `src/training/__init__.py` (vide), `src/training/callbacks.py`, `tests/training/__init__.py` (vide), `tests/training/test_callbacks.py`
- Delete: `src/models/callbacks.py`, `tests/models/test_callbacks.py`
- Modify: `src/models/base_model.py` (import), `src/train.py` (import)

**Interfaces:**
- Consumes: rien (hors `torch`).
- Produces: `src.training.callbacks.EarlyStopping` — API **identique** à l'actuelle (`__init__(monitor, mode, patience, min_delta, restore_best_weights)`, `step(epoch_metrics, model) -> bool`, `restore(model)`, attributs `best_score`, `best_epoch`, `stopped_epoch`).

- [ ] **Step 1: Déplacer le module**

```bash
git mv src/models/callbacks.py src/training/callbacks.py
touch src/training/__init__.py tests/training/__init__.py
git mv tests/models/test_callbacks.py tests/training/test_callbacks.py
```

- [ ] **Step 2: Mettre à jour l'import du test**

Dans `tests/training/test_callbacks.py` : `from src.models.callbacks import EarlyStopping` → `from src.training.callbacks import EarlyStopping`. Aucune autre modification.

- [ ] **Step 3: Mettre à jour les imports amont**

- `src/models/base_model.py` : `from src.models.callbacks import ...` — vérifier : le fichier n'importe PAS `EarlyStopping` directement (il reçoit l'instance en argument). Aucun changement probable ; si `grep` trouve un import, le rediriger.
- `src/train.py` : `from src.models.callbacks import EarlyStopping` → `from src.training.callbacks import EarlyStopping`.

Run: `grep -rn "models.callbacks" src tests`
Expected: plus aucune occurrence après correction.

- [ ] **Step 4: Vérifier tests**

Run: `ruff check src/training tests/training && /Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`
Expected: PASS (mêmes tests, `test_callbacks.py` sous son nouveau chemin)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: deplacer EarlyStopping dans src/training/callbacks.py"
```

---

## Task 11: `src/training/components.py`

**Files:**
- Create: `src/training/components.py`, `tests/training/test_components.py`

**Interfaces:**
- Consumes: `src.config` (`OptimConfig`, `SchedulerConfig`, `TrainConfig`, `DataConfig`, `TransformConfig`), `src.tasks.TaskSpec`, `src.features` (`build_transform`, `compute_dataset_mean_std`).
- Produces:
  - `resolve_device(name: str) -> torch.device`
  - `split_off_test(df: pd.DataFrame, data_cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]` — `train_test_split` stratifié sur `data_cfg.stratify` (`None` si `"no"`).
  - `make_splits(train_df: pd.DataFrame, train_cfg: TrainConfig, data_cfg: DataConfig) -> Iterator[tuple[pd.DataFrame, pd.DataFrame | None]]` — `none` → 1×`(df, None)` ; `holdout` → 1×`(tr, va)` stratifié `val_size` ; `cross` → `cross_val_folds`× via `StratifiedKFold` (sur `data_cfg.stratify`, sinon `KFold`).
  - `TransformBuilder(transform_cfg: TransformConfig, task: TaskSpec, reinhard_normalizer=None)` avec `.build(fit_df: pd.DataFrame, *, training: bool) -> v2.Compose` — calcule mean/std via `compute_dataset_mean_std(fit_df, transform_cfg, task, reinhard_normalizer)` puis `build_transform(...)`.
  - `build_optimizer(optim_cfg: OptimConfig, params) -> torch.optim.Optimizer`
  - `build_scheduler(sched_cfg: SchedulerConfig, optimizer, *, nb_epoch: int)` — `None` / `ReduceLROnPlateau` / `CosineAnnealingLR` / `StepLR` comme spec §6.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/training/test_components.py` :
```python
"""Tests des composants d'entraînement (splits, device, optim, scheduler, transforms)."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from src.config import DataConfig, OptimConfig, SchedulerConfig, TrainConfig, TransformConfig
from src.tasks import get_task
from src.training.components import (
    TransformBuilder,
    build_optimizer,
    build_scheduler,
    make_splits,
    resolve_device,
    split_off_test,
)


def _df(n=40):
    return pd.DataFrame(
        {"label": ([0, 1] * (n // 2)), "type_dataset": (["a", "b"] * (n // 2)), "dataset": "barcelona"}
    )


def test_resolve_device_returns_torch_device():
    assert isinstance(resolve_device("cpu"), torch.device)
    assert isinstance(resolve_device("auto"), torch.device)


def test_split_off_test_ratio():
    tr, te = split_off_test(_df(100), DataConfig(test_size=0.2, stratify="type_dataset"))
    assert len(te) == 20
    assert len(tr) == 80


def test_make_splits_holdout_yields_one_pair():
    splits = list(make_splits(_df(), TrainConfig(validation="holdout", val_size=0.25), DataConfig()))
    assert len(splits) == 1
    tr, va = splits[0]
    assert va is not None
    assert len(va) == 10


def test_make_splits_none_yields_no_validation():
    splits = list(make_splits(_df(), TrainConfig(validation="none"), DataConfig()))
    assert len(splits) == 1
    assert splits[0][1] is None


def test_make_splits_cross_yields_k_folds():
    splits = list(
        make_splits(_df(60), TrainConfig(validation="cross", cross_val_folds=3), DataConfig())
    )
    assert len(splits) == 3
    for tr, va in splits:
        assert va is not None and len(tr) + len(va) == 60


def test_build_optimizer_dispatch():
    params = [torch.nn.Parameter(torch.randn(2))]
    assert build_optimizer(OptimConfig(name="adam", lr=1e-3), params).defaults["lr"] == 1e-3
    assert type(build_optimizer(OptimConfig(name="sgd"), params)).__name__ == "SGD"


def test_build_scheduler_dispatch():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.randn(2))], lr=0.1)
    assert build_scheduler(SchedulerConfig(name="none"), opt, nb_epoch=5) is None
    assert isinstance(build_scheduler(SchedulerConfig(name="plateau"), opt, nb_epoch=5), ReduceLROnPlateau)
    assert isinstance(build_scheduler(SchedulerConfig(name="cosine"), opt, nb_epoch=5), CosineAnnealingLR)
    assert isinstance(build_scheduler(SchedulerConfig(name="step"), opt, nb_epoch=6), StepLR)


def test_transform_builder_produces_compose(tmp_path):
    rows = []
    for i in range(6):
        p = tmp_path / f"i{i}.png"
        Image.fromarray(np.uint8(np.random.rand(24, 24, 3) * 255)).save(p)
        rows.append({"path": str(p), "label": i % 2, "dataset": "barcelona"})
    df = pd.DataFrame(rows)
    tb = TransformBuilder(TransformConfig(reinhard=False, clahe=False, jpeg=False), get_task("type"))
    compose = tb.build(df, training=False)
    out = compose(torch.randint(0, 255, (3, 24, 24), dtype=torch.uint8))
    assert out.shape == (3, 224, 224)
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/training/test_components.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.training.components'`

- [ ] **Step 3: Implémenter `src/training/components.py`**

```python
"""Composants assemblés par le runner : device, splits, transforms, optim, scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from src.features.stats import compute_dataset_mean_std
from src.features.transform import build_transform

if TYPE_CHECKING:
    import pandas as pd
    from torchvision.transforms import v2

    from src.config import DataConfig, OptimConfig, SchedulerConfig, TrainConfig, TransformConfig
    from src.tasks import TaskSpec


def resolve_device(name: str) -> torch.device:
    """``"auto"`` -> cuda / mps / cpu selon disponibilité."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _strat_key(df: pd.DataFrame, data_cfg: DataConfig):
    return None if data_cfg.stratify == "no" else df[data_cfg.stratify]


def split_off_test(df: pd.DataFrame, data_cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare un jeu de test stratifié."""
    return train_test_split(
        df, test_size=data_cfg.test_size, stratify=_strat_key(df, data_cfg)
    )


def make_splits(
    train_df: pd.DataFrame, train_cfg: TrainConfig, data_cfg: DataConfig
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame | None]]:
    """Itère les paires ``(train, val)`` selon le mode de validation."""
    if train_cfg.validation == "none":
        yield train_df, None
        return
    if train_cfg.validation == "holdout":
        tr, va = train_test_split(
            train_df, test_size=train_cfg.val_size, stratify=_strat_key(train_df, data_cfg)
        )
        yield tr, va
        return
    # cross
    if data_cfg.stratify == "no":
        splitter = KFold(n_splits=train_cfg.cross_val_folds, shuffle=True)
        indices = splitter.split(train_df)
    else:
        splitter = StratifiedKFold(n_splits=train_cfg.cross_val_folds, shuffle=True)
        indices = splitter.split(train_df, train_df[data_cfg.stratify])
    for train_idx, val_idx in indices:
        yield train_df.iloc[train_idx], train_df.iloc[val_idx]


class TransformBuilder:
    """Fabrique de transforms : calcule mean/std sur le sous-ensemble d'entraînement fourni."""

    def __init__(self, transform_cfg: TransformConfig, task: TaskSpec, reinhard_normalizer=None):
        self.cfg = transform_cfg
        self.task = task
        self.reinhard_normalizer = reinhard_normalizer

    def build(self, fit_df: pd.DataFrame, *, training: bool) -> v2.Compose:
        mean, std = compute_dataset_mean_std(
            fit_df, self.cfg, self.task, reinhard_normalizer=self.reinhard_normalizer
        )
        return build_transform(
            self.cfg,
            training=training,
            reinhard_normalizer=self.reinhard_normalizer,
            dataset_mean=mean,
            dataset_std=std,
        )


def build_optimizer(optim_cfg: OptimConfig, params) -> torch.optim.Optimizer:
    """Adam ou SGD selon la config."""
    if optim_cfg.name == "adam":
        return Adam(params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    if optim_cfg.name == "sgd":
        return SGD(params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    raise ValueError(f"optimizer inconnu : {optim_cfg.name!r}")


def build_scheduler(sched_cfg: SchedulerConfig, optimizer, *, nb_epoch: int):
    """Scheduler de learning rate, ou ``None``."""
    if sched_cfg.name == "none":
        return None
    if sched_cfg.name == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", patience=sched_cfg.patience, min_lr=sched_cfg.lr_min
        )
    if sched_cfg.name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=nb_epoch, eta_min=sched_cfg.lr_min)
    if sched_cfg.name == "step":
        return StepLR(optimizer, step_size=max(1, nb_epoch // 3), gamma=sched_cfg.step_gamma)
    raise ValueError(f"scheduler inconnu : {sched_cfg.name!r}")
```

- [ ] **Step 4: Vérifier tests**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/training/test_components.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Lint + suite + commit**

```bash
ruff check src/training/components.py tests/training/test_components.py
/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q
git add src/training/components.py tests/training/test_components.py
git commit -m "feat: composants d'entrainement (device, splits, transforms, optim, scheduler)"
```

---

## Task 12: `src/training/trainer.py` + suppression de `base_model.py`

**Files:**
- Create: `src/training/trainer.py`, `tests/training/test_trainer.py`
- Delete: `src/models/base_model.py`, `tests/models/test_train_loop.py`

**Interfaces:**
- Consumes: `src.config.TrainConfig`, `src.features.dataset.DatasetTransform` (via DataLoader — en fait juste des `Dataset`), `src.training.callbacks.EarlyStopping`.
- Produces:
  - `Trainer(model, config: TrainConfig, criterion, *, optimizer, scheduler=None, metrics=None, callbacks=(), device="cpu")`
  - `Trainer.fit(train_ds, val_ds=None) -> dict[str, list]` — history (`defaultdict(list)` matérialisé en `dict`). Clés : `train_loss`, `lr`, métriques `train_*`, et si `val_ds` : `val_loss`, `val_*`.
  - `Trainer.evaluate(test_ds, metrics=None) -> dict` — comme `BaseModel.evaluate` (running loss + collection `test_`).
  - `Trainer.predict(x)`
  - Garde-fou : `fit` lève `ValueError` (message contenant "validation") si `val_ds is None` et (`scheduler` est un `ReduceLROnPlateau` **ou** un callback est présent).

- [ ] **Step 1: Écrire les tests qui échouent** (portage de `test_train_loop.py` + `evaluate`)

`tests/training/test_trainer.py` :
```python
"""Tests de la boucle d'entraînement (Trainer)."""

import pytest
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import TensorDataset

from src.config import TrainConfig
from src.training.callbacks import EarlyStopping
from src.training.trainer import Trainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def _ds(n=16):
    return TensorDataset(torch.randn(n, 4), torch.randint(0, 2, (n,)))


def _trainer(model, *, scheduler=None, callbacks=(), nb_epoch=3, lr=0.1, batch_size=4):
    cfg = TrainConfig(nb_epoch=nb_epoch, batch_size=batch_size, validation="holdout")
    opt = SGD(model.parameters(), lr=lr)
    return Trainer(
        model,
        cfg,
        nn.CrossEntropyLoss(),
        optimizer=opt,
        scheduler=scheduler(opt) if scheduler else None,
        callbacks=callbacks,
        device="cpu",
    )


def test_step_scheduler_advances_once_per_epoch():
    model = TinyModel()
    opt = SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(opt, step_size=1, gamma=0.5)
    t = Trainer(model, TrainConfig(nb_epoch=3, batch_size=4), nn.CrossEntropyLoss(),
                optimizer=opt, scheduler=scheduler, device="cpu")
    t.fit(_ds(), _ds())
    assert scheduler.last_epoch == 3
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5**3)


def test_plateau_scheduler_receives_val_loss():
    model = TinyModel()
    opt = SGD(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(opt, mode="min", patience=0, factor=0.5)
    t = Trainer(model, TrainConfig(nb_epoch=4, batch_size=4), nn.CrossEntropyLoss(),
                optimizer=opt, scheduler=scheduler, device="cpu")
    history = t.fit(_ds(), _ds())
    assert scheduler.best == min(history["val_loss"][: scheduler.last_epoch + 1])


def test_lr_logged_in_history():
    history = _trainer(TinyModel(), scheduler=lambda o: StepLR(o, step_size=1)).fit(_ds(), _ds())
    assert "lr" in history and len(history["lr"]) == 3


def test_early_stopping_breaks_and_restores():
    class StopAfterSecond(EarlyStopping):
        restore_calls = 0

        def step(self, m, model):
            super().step(m, model)
            return self._epoch >= 1

        def restore(self, model):
            type(self).restore_calls += 1
            super().restore(model)

    model = TinyModel()
    with torch.no_grad():
        model.linear.weight.fill_(0.123)
    frozen = model.linear.weight.clone()
    es = StopAfterSecond(monitor="val_loss", patience=99)
    history = _trainer(model, callbacks=[es], nb_epoch=10, lr=0.0).fit(_ds(), _ds())
    assert len(history["train_loss"]) == 2
    assert StopAfterSecond.restore_calls == 1
    assert torch.equal(model.linear.weight, frozen)


def test_early_stopping_without_validation_raises():
    with pytest.raises(ValueError, match="validation"):
        _trainer(TinyModel(), callbacks=[EarlyStopping()]).fit(_ds())


def test_plateau_without_validation_raises():
    with pytest.raises(ValueError, match="validation"):
        _trainer(TinyModel(), scheduler=lambda o: ReduceLROnPlateau(o)).fit(_ds())


def test_evaluate_returns_test_loss():
    from torchmetrics import MetricCollection
    from torchmetrics.classification import MulticlassAccuracy

    model = TinyModel()
    t = _trainer(model)
    t.fit(_ds(), _ds())
    out = t.evaluate(_ds(), metrics=MetricCollection({"acc": MulticlassAccuracy(num_classes=2)}))
    assert "test_loss" in out
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/training/test_trainer.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.training.trainer'`

- [ ] **Step 3: Implémenter `src/training/trainer.py`** (portage de `BaseModel`)

```python
"""Boucle d'entraînement / validation / évaluation, isolée du modèle."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from torch import nn
    from torchmetrics import MetricCollection

    from src.config import TrainConfig


class Trainer:
    """Entraîne un ``nn.Module`` : boucle epoch, scheduler, callbacks, métriques."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        criterion,
        *,
        optimizer,
        scheduler=None,
        metrics: MetricCollection | None = None,
        callbacks=(),
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = list(callbacks)
        self.device = torch.device(device)
        self.train_metrics = metrics.clone(prefix="train_") if metrics is not None else None
        self.val_metrics = metrics.clone(prefix="val_") if metrics is not None else None
        self.test_metrics = None

    def fit(self, train_ds, val_ds=None) -> dict:
        self.model.to(self.device)
        if self.train_metrics is not None:
            self.train_metrics = self.train_metrics.to(self.device)
            self.val_metrics = self.val_metrics.to(self.device)

        is_plateau = isinstance(self.scheduler, ReduceLROnPlateau)
        if val_ds is None and (is_plateau or self.callbacks):
            raise ValueError(
                "early_stopping et ReduceLROnPlateau ont besoin d'un jeu de validation."
            )

        train_loader = DataLoader(train_ds, self.config.batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_ds, self.config.batch_size, shuffle=False) if val_ds is not None else None
        )

        history: dict = defaultdict(list)
        for epoch in range(self.config.nb_epoch):
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}: ", unit="batch") as bar:
                metrics_train = self._train_epoch(train_loader, bar)
                history["lr"].append(self.optimizer.param_groups[0]["lr"])
                metrics_val = self._validate_epoch(val_loader, bar, metrics_train) if val_loader else {}

            epoch_metrics = {**metrics_train, **metrics_val}
            for key, value in epoch_metrics.items():
                history[key].append(value)

            if self.scheduler is not None:
                if is_plateau:
                    self.scheduler.step(epoch_metrics["val_loss"])
                else:
                    self.scheduler.step()

            if any(cb.step(epoch_metrics, self.model) for cb in self.callbacks):
                print(f"Arrêt anticipé à l'epoch {epoch + 1}")
                break

        for cb in self.callbacks:
            cb.restore(self.model)
        return dict(history)

    def predict(self, x):
        return self.model(x)

    def _train_epoch(self, loader, bar) -> dict:
        self.model.train()
        n_samples, running = 0, 0.0
        out = OrderedDict({"train_loss": 0.0})
        for x, y in loader:
            bar.update(1)
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(x)
            loss = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()

            bs = x.size(0)
            n_samples += bs
            running += loss.item() * bs
            out["train_loss"] = running / n_samples
            if self.train_metrics is not None:
                self.train_metrics.update(preds.detach(), y)
                out.update({k: v.item() for k, v in self.train_metrics.compute().items()})
            bar.set_postfix(out)
        if self.train_metrics is not None:
            self.train_metrics.reset()
        return dict(out)

    @torch.no_grad()
    def _validate_epoch(self, loader, bar, metrics_train) -> dict:
        self.model.eval()
        n_samples, running = 0, 0.0
        out = OrderedDict({"val_loss": 0.0})
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x)
            loss = self.criterion(preds, y)
            bs = x.size(0)
            n_samples += bs
            running += loss.item() * bs
            out["val_loss"] = running / n_samples
            if self.val_metrics is not None:
                self.val_metrics.update(preds.detach(), y)
                out.update({k: v.item() for k, v in self.val_metrics.compute().items()})
            bar.set_postfix({**metrics_train, **out})
        if self.val_metrics is not None:
            self.val_metrics.reset()
        return dict(out)

    @torch.no_grad()
    def evaluate(self, test_ds, metrics: MetricCollection | None = None) -> dict:
        self.model.to(self.device)
        self.model.eval()
        test_metrics = metrics.clone(prefix="test_").to(self.device) if metrics is not None else None
        loader = DataLoader(test_ds, self.config.batch_size, shuffle=False)

        n_samples, running = 0, 0.0
        out = OrderedDict({"test_loss": 0.0})
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x)
            loss = self.criterion(preds, y)
            bs = x.size(0)
            n_samples += bs
            running += loss.item() * bs
            out["test_loss"] = running / n_samples
            if test_metrics is not None:
                test_metrics.update(preds.detach(), y)
                out.update(
                    {
                        k: (v.item() if v.numel() == 1 else v.tolist())
                        for k, v in test_metrics.compute().items()
                    }
                )
        return dict(out)
```

Note : le test `test_early_stopping_breaks_and_restores` attend `restore` appelé **une** fois par callback en fin de `fit`. L'`EarlyStopping.restore` réel est idempotent (recharge `_best_state` si présent). OK.

- [ ] **Step 4: Vérifier les tests du Trainer**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/training/test_trainer.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Supprimer `base_model.py` et l'ancien test**

```bash
git rm src/models/base_model.py tests/models/test_train_loop.py
grep -rn "base_model" src tests
```
Expected grep: plus aucune occurrence. Corriger `src/models/dataset_model.py` (déjà fait Task 7) et `src/train.py` (`model.fit(...)` — `train.py` sera supprimé Task 17 ; s'il casse l'import du module au `pytest` collect, l'entourer d'un `try` n'est pas souhaitable : vérifier que rien n'importe `src.train`). 

Run: `grep -rn "import.*src.train\b\|from src.train " tests`
Expected: aucune occurrence (aucun test n'importe `src/train.py`).

- [ ] **Step 6: Lint + suite complète**

Run: `ruff check src/training tests/training && /Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`
Expected: PASS. Compter : les tests `test_train_loop.py` (6) sont remplacés par `test_trainer.py` (7).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: Trainer isole + suppression de BaseModel"
```

---

## Task 13: `src/experiments/artifacts.py`

**Files:**
- Create: `src/experiments/__init__.py` (vide), `src/experiments/artifacts.py`, `tests/experiments/__init__.py` (vide), `tests/experiments/test_artifacts.py`

**Interfaces:**
- Consumes: `src.config.ExperimentConfig`, `src.models.helpers.plot_train_metrics`, `src.models.metrics.display_metrics`, `src.tasks.TaskSpec`.
- Produces:
  - `@dataclass RunResult` : `config: ExperimentConfig`, `task: TaskSpec`, `histories: list[dict]`, `test_metrics: dict`, `test_metrics_folds: list[dict]`, `best_state: dict`, `run_dir: Path | None`, `duration_s: float`, `status: str = "ok"`, `error: str = ""`.
  - `flatten(d: dict, prefix="") -> dict` — `{"a": {"b": 1}}` → `{"a.b": 1}`.
  - `dirname_for(config: ExperimentConfig) -> str` — `"<YYYYMMDD-HHMMSS>_<hash8>"`, hash = SHA-256 de `json.dumps(config.model_dump(mode="json"), sort_keys=True)`.
  - `write_run_dir(result: RunResult, root: Path) -> Path` — crée le dossier, écrit `config.yaml`, `history.json`, `metrics_test.json`, `figures/` (epoch via `plot_train_metrics` sauf CV ; test via `display_metrics`), `checkpoint.pt` si `config.output.save_checkpoint`. Renseigne `result.run_dir`.
  - `append_leaderboard(path: Path, result: RunResult) -> None` — append une ligne (crée l'entête si absent).

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/experiments/test_artifacts.py` :
```python
"""Tests des artefacts de run (dossier + leaderboard)."""

import json

import pandas as pd
import pytest
import torch

from src.config import ExperimentConfig, load_config
from src.experiments.artifacts import (
    RunResult,
    append_leaderboard,
    dirname_for,
    flatten,
    write_run_dir,
)
from src.tasks import get_task


def test_flatten_nested_dict():
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}


def test_dirname_is_deterministic_for_same_config():
    cfg = ExperimentConfig()
    a, b = dirname_for(cfg), dirname_for(cfg)
    assert a.split("_")[1] == b.split("_")[1]  # même hash


def _result(tmp_path, validation="holdout"):
    cfg = ExperimentConfig(train={"validation": validation, "nb_epoch": 2})
    hist = {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9], "lr": [0.01, 0.01]}
    test_metrics = {"test_loss": 0.7, "test_f1_macro": 0.5, "test_mcc": 0.4}
    return RunResult(
        config=cfg, task=get_task("type"), histories=[hist],
        test_metrics=test_metrics, test_metrics_folds=[test_metrics],
        best_state={"w": torch.zeros(1)}, run_dir=None, duration_s=1.0,
    )


def test_write_run_dir_creates_expected_files(tmp_path):
    result = _result(tmp_path)
    run_dir = write_run_dir(result, tmp_path)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "metrics_test.json").exists()
    assert (run_dir / "checkpoint.pt").exists()
    reloaded = load_config(run_dir / "config.yaml")
    assert reloaded.train.nb_epoch == 2


def test_append_leaderboard_adds_rows_without_duplicating_header(tmp_path):
    lb = tmp_path / "leaderboard.csv"
    append_leaderboard(lb, _result(tmp_path))
    append_leaderboard(lb, _result(tmp_path))
    df = pd.read_csv(lb)
    assert len(df) == 2
    assert "loss.name" in df.columns
    assert "test_f1_macro" in df.columns
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/experiments/test_artifacts.py -q`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implémenter `src/experiments/artifacts.py`**

```python
"""Écriture des artefacts d'un run : dossier horodaté + leaderboard cumulatif."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
import yaml

from src.models.helpers import plot_train_metrics
from src.models.metrics import display_metrics

if TYPE_CHECKING:
    from src.config import ExperimentConfig
    from src.tasks import TaskSpec


@dataclass
class RunResult:
    config: ExperimentConfig
    task: TaskSpec
    histories: list[dict]
    test_metrics: dict
    test_metrics_folds: list[dict]
    best_state: dict
    run_dir: Path | None = None
    duration_s: float = 0.0
    status: str = "ok"
    error: str = ""


def flatten(d: dict, prefix: str = "") -> dict:
    """Aplati un dict imbriqué : ``{"a": {"b": 1}}`` -> ``{"a.b": 1}``."""
    out: dict = {}
    for key, value in d.items():
        composed = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{composed}."))
        else:
            out[composed] = value
    return out


def _config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def dirname_for(config: ExperimentConfig) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}_{_config_hash(config)}"


def write_run_dir(result: RunResult, root: Path) -> Path:
    """Écrit config / history / métriques test / figures / checkpoint. Renvoie le dossier."""
    root = Path(root)
    run_dir = root / dirname_for(result.config)
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    result.run_dir = run_dir

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(result.config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    )

    hist_payload = result.histories[0] if len(result.histories) == 1 else result.histories
    (run_dir / "history.json").write_text(json.dumps(hist_payload, indent=2))
    (run_dir / "metrics_test.json").write_text(json.dumps(result.test_metrics, indent=2, default=str))

    if result.config.output.save_figures:
        is_cv = result.config.train.validation == "cross"
        if not is_cv and result.histories:
            plot_train_metrics(result.histories[0], save_dir=figures)
        # figures test : reconstruites depuis la MetricCollection déjà calculée en amont ?
        # ici on ne reçoit que des scalaires ; le runner passe la collection compute()e
        # via result.test_metrics_folds[0] au format display_metrics -> voir Task 14.

    if result.config.output.save_checkpoint:
        torch.save(result.best_state, run_dir / "checkpoint.pt")

    return run_dir


def append_leaderboard(path: Path, result: RunResult) -> None:
    """Ajoute une ligne au CSV cumulatif (entête créé si le fichier est absent)."""
    path = Path(path)
    row = {
        "run_dir": str(result.run_dir) if result.run_dir else "",
        "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "task": result.config.task,
        "seed": result.config.seed,
        "duration_s": round(result.duration_s, 1),
        "status": result.status,
        "error": result.error,
    }
    row.update(flatten(result.config.model_dump(mode="json")))
    row.update({k if k.startswith("test_") else f"test_{k}": v for k, v in result.test_metrics.items()})
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
```

Note : la génération des **figures test** (`display_metrics`) exige la collection torchmetrics `.compute()` (avec tenseurs `*_per_class` et `confusion_matrix`), pas les scalaires aplatis. Le runner (Task 14) fournira ce dict brut ; ici on ajoute un champ optionnel :

Ajouter à `RunResult` : `test_metrics_raw: dict | None = None` (le dict `MetricCollection.compute()` du fold 0, préfixé `test_`). Et dans `write_run_dir`, après le bloc `plot_train_metrics` :
```python
        if result.test_metrics_raw is not None:
            display_metrics(
                result.test_metrics_raw,
                num_classes=result.task.num_classes,
                title="Test",
                save_dir=figures,
                verbose=False,
            )
```
Mettre à jour le test `_result` pour passer `test_metrics_raw=None` (défaut) — les assertions n'en dépendent pas.

- [ ] **Step 4: Vérifier tests**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/experiments/test_artifacts.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint + suite + commit**

```bash
ruff check src/experiments/artifacts.py tests/experiments
/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q
git add src/experiments/__init__.py src/experiments/artifacts.py tests/experiments/__init__.py tests/experiments/test_artifacts.py
git commit -m "feat: artefacts de run (dossier horodate + leaderboard)"
```

---

## Task 14: `src/experiments/runner.py`

**Files:**
- Create: `src/experiments/runner.py`, `tests/experiments/test_runner_smoke.py`

**Interfaces:**
- Consumes: tout ce qui précède.
- Produces:
  - `seed_everything(seed: int) -> None` — `random`, `numpy`, `torch`, `torch.cuda`.
  - `run_experiment(config: ExperimentConfig, output_root: Path | None = None) -> RunResult`
    - `output_root` par défaut `config.output.root`.
  - `_warn_if_missing_classes(df, task) -> None` — `warnings.warn` si `df[task.label_column].nunique() < task.num_classes`.
  - `_assemble_result(config, task, histories, fold_test_metrics, fold_raw_metrics, models, duration) -> RunResult` — agrège : holdout → métriques du fold unique ; cross → `{"<k>_mean": mean, "<k>_std": std}` pour chaque scalaire, `best_state` = `models[0].state_dict()`, `test_metrics_raw` = `fold_raw_metrics[0]`.

- [ ] **Step 1: Écrire le test smoke qui échoue**

`tests/experiments/test_runner_smoke.py` :
```python
"""Test d'intégration léger du runner (nécessite les données sur disque)."""

import pandas as pd
import pytest

from src.config import ExperimentConfig
from src.constants import DATASET_PATHS
from src.experiments.runner import run_experiment

pytestmark = pytest.mark.skipif(
    not DATASET_PATHS["barcelona"].exists(), reason="données absentes"
)


def test_holdout_run_produces_artifacts(tmp_path):
    cfg = ExperimentConfig.model_validate(
        {
            "task": "type",
            "seed": 0,
            "data": {"n_images": 60, "stratify": "type", "test_size": 0.25},
            "transform": {"reinhard": False, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 1, "batch_size": 16, "validation": "holdout", "val_size": 0.25},
            "output": {"root": str(tmp_path), "save_checkpoint": True},
        }
    )
    result = run_experiment(cfg, output_root=tmp_path)
    assert result.status == "ok"
    assert result.run_dir.exists()
    assert (result.run_dir / "config.yaml").exists()
    assert (result.run_dir / "metrics_test.json").exists()
    assert "test_loss" in result.test_metrics
    assert (tmp_path / "leaderboard.csv").exists()
    assert len(pd.read_csv(tmp_path / "leaderboard.csv")) == 1
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/experiments/test_runner_smoke.py -q`
Expected: FAIL `ModuleNotFoundError` (ou `skip` si pas de données — dans ce cas, valider la logique via revue + Task 16 manuelle).

- [ ] **Step 3: Implémenter `src/experiments/runner.py`**

```python
"""Exécution d'une expérience unique : config -> entraînement -> artefacts."""

from __future__ import annotations

import random
import statistics
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.experiments.artifacts import RunResult, append_leaderboard, write_run_dir
from src.features.dataframe import create_dataframe
from src.features.dataset import BloodCellDataset, DatasetTransform
from src.features.stats import build_reinhard
from src.models import build_model
from src.models.losses import build_loss
from src.models.metrics import build_metrics
from src.tasks import get_task
from src.training.callbacks import EarlyStopping
from src.training.components import (
    TransformBuilder,
    build_optimizer,
    build_scheduler,
    make_splits,
    resolve_device,
    split_off_test,
)
from src.training.trainer import Trainer

if TYPE_CHECKING:
    from src.config import ExperimentConfig
    from src.tasks import TaskSpec


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warn_if_missing_classes(df, task: TaskSpec) -> None:
    present = df[task.label_column].nunique()
    if present < task.num_classes:
        warnings.warn(
            f"{task.name}: {present}/{task.num_classes} classes présentes dans le DataFrame.",
            stacklevel=2,
        )


def _aggregate(fold_metrics: list[dict]) -> dict:
    if len(fold_metrics) == 1:
        return dict(fold_metrics[0])
    keys = fold_metrics[0]
    out: dict = {}
    for key in keys:
        values = [m[key] for m in fold_metrics if isinstance(m.get(key), (int, float))]
        if values:
            out[f"{key}_mean"] = statistics.fmean(values)
            out[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return out


def run_experiment(config: ExperimentConfig, output_root: Path | None = None) -> RunResult:
    """Lance une expérience, écrit son dossier + une ligne de leaderboard."""
    started = time.time()
    root = Path(output_root) if output_root is not None else config.output.root
    seed_everything(config.seed)
    task = get_task(config.task)
    device = resolve_device(config.train.device)

    df = create_dataframe(config.data, task)
    _warn_if_missing_classes(df, task)
    train_df, test_df = split_off_test(df, config.data)

    normalizer = build_reinhard(train_df, config.transform) if config.transform.reinhard else None
    tb = TransformBuilder(config.transform, task, reinhard_normalizer=normalizer)
    test_ds = DatasetTransform(BloodCellDataset(test_df, task), tb.build(train_df, training=False))

    histories: list[dict] = []
    fold_test_scalar: list[dict] = []
    fold_test_raw: list[dict] = []
    models: list = []

    for tr_df, va_df in make_splits(train_df, config.train, config.data):
        model = build_model(config.model, task).to(device)
        criterion = build_loss(config.loss, task, tr_df).to(device)
        optimizer = build_optimizer(config.optim, model.parameters())
        scheduler = build_scheduler(config.scheduler, optimizer, nb_epoch=config.train.nb_epoch)
        callbacks = (
            [EarlyStopping(monitor="val_loss", mode="min", patience=config.train.early_stopping_patience)]
            if config.train.early_stopping_patience
            else []
        )

        train_ds = DatasetTransform(BloodCellDataset(tr_df, task), tb.build(tr_df, training=True))
        val_ds = (
            DatasetTransform(BloodCellDataset(va_df, task), tb.build(tr_df, training=False))
            if va_df is not None
            else None
        )

        trainer = Trainer(
            model, config.train, criterion,
            optimizer=optimizer, scheduler=scheduler,
            metrics=build_metrics(task), callbacks=callbacks, device=device,
        )
        histories.append(trainer.fit(train_ds, val_ds))

        test_collection = build_metrics(task, for_test=True, device=device)
        scalars = trainer.evaluate(test_ds, metrics=test_collection)
        fold_test_scalar.append({k: v for k, v in scalars.items() if isinstance(v, (int, float))})
        fold_test_raw.append({f"test_{k}": v for k, v in test_collection.compute().items()})
        models.append(model)

    duration = time.time() - started
    result = RunResult(
        config=config,
        task=task,
        histories=histories,
        test_metrics=_aggregate(fold_test_scalar),
        test_metrics_folds=fold_test_scalar,
        best_state=models[0].state_dict(),
        duration_s=duration,
    )
    result.test_metrics_raw = fold_test_raw[0]

    write_run_dir(result, root)
    append_leaderboard(root / "leaderboard.csv", result)
    return result
```

Note : `RunResult` doit accepter l'attribut `test_metrics_raw` — ajouté en Task 13 comme `field` avec défaut `None`.

- [ ] **Step 4: Vérifier tests**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/experiments/test_runner_smoke.py -q`
Expected: PASS si données présentes, sinon `skip`.

- [ ] **Step 5: Lint + suite + commit**

```bash
ruff check src/experiments/runner.py tests/experiments/test_runner_smoke.py
/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q
git add src/experiments/runner.py tests/experiments/test_runner_smoke.py src/experiments/artifacts.py
git commit -m "feat: runner d'experience unique"
```

---

## Task 15: `src/experiments/grid.py`

**Files:**
- Create: `src/experiments/grid.py`, `tests/experiments/test_grid.py`

**Interfaces:**
- Consumes: `src.config` (`ExperimentConfig`, `deep_merge`, `parse_overrides`), `src.experiments.runner.run_experiment`, `sklearn.model_selection.ParameterGrid`.
- Produces:
  - `_apply_overrides(base: ExperimentConfig, flat: dict) -> ExperimentConfig` — `flat` à clés pointées (`{"loss.name": "focal"}`) → dict imbriqué → `deep_merge` sur `base.model_dump(mode="json")` → `ExperimentConfig.model_validate`.
  - `expand_grid(base: ExperimentConfig, grid_file: Path) -> list[ExperimentConfig]` — YAML (dict de listes, ou liste de dicts de listes) → `ParameterGrid` → configs.
  - `run_grid(base: ExperimentConfig, grid_file: Path, *, task: str | None = None, resume: bool = False, output_root: Path | None = None) -> None`.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/experiments/test_grid.py` :
```python
"""Tests du dépliage de grille."""

import pytest
from pydantic import ValidationError

from src.config import ExperimentConfig
from src.experiments.grid import _apply_overrides, expand_grid


def test_apply_overrides_returns_validated_config():
    cfg = _apply_overrides(ExperimentConfig(), {"loss.name": "focal", "train.nb_epoch": 5})
    assert cfg.loss.name == "focal"
    assert cfg.train.nb_epoch == 5


def test_apply_overrides_rejects_invalid_combo():
    with pytest.raises(ValidationError):
        _apply_overrides(ExperimentConfig(), {"loss.name": "not_a_loss"})


def test_expand_grid_cartesian_product(tmp_path):
    grid = tmp_path / "g.yaml"
    grid.write_text("loss.name: [cross_entropy, focal]\nloss.beta: [0.99, 0.999]\n")
    configs = expand_grid(ExperimentConfig(), grid)
    assert len(configs) == 4
    combos = {(c.loss.name, c.loss.beta) for c in configs}
    assert combos == {("cross_entropy", 0.99), ("cross_entropy", 0.999),
                      ("focal", 0.99), ("focal", 0.999)}


def test_expand_grid_list_of_dicts(tmp_path):
    grid = tmp_path / "g.yaml"
    grid.write_text(
        "- loss.name: [cross_entropy]\n"
        "- loss.name: [focal]\n"
        "  loss.gamma: [1.0, 2.0]\n"
    )
    configs = expand_grid(ExperimentConfig(), grid)
    assert len(configs) == 3
```

- [ ] **Step 2: Vérifier l'échec**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/experiments/test_grid.py -q`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implémenter `src/experiments/grid.py`**

```python
"""Grid search : déplie une grille de surcharges en configs et lance chaque run."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from sklearn.model_selection import ParameterGrid

from src.config import ExperimentConfig, deep_merge
from src.experiments.artifacts import RunResult, append_leaderboard, dirname_for
from src.experiments.runner import run_experiment
from src.tasks import get_task

if TYPE_CHECKING:
    pass


def _nest(flat: dict) -> dict:
    """``{"a.b": 1}`` -> ``{"a": {"b": 1}}``."""
    out: dict = {}
    for dotted, value in flat.items():
        node = out
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return out


def _apply_overrides(base: ExperimentConfig, flat: dict) -> ExperimentConfig:
    merged = deep_merge(base.model_dump(mode="json"), _nest(flat))
    return ExperimentConfig.model_validate(merged)


def expand_grid(base: ExperimentConfig, grid_file: Path) -> list[ExperimentConfig]:
    raw = yaml.safe_load(Path(grid_file).read_text())
    return [_apply_overrides(base, combo) for combo in ParameterGrid(raw)]


def run_grid(
    base: ExperimentConfig,
    grid_file: Path,
    *,
    task: str | None = None,
    resume: bool = False,
    output_root: Path | None = None,
) -> None:
    if task:
        base = _apply_overrides(base, {"task": task})
        get_task(task)  # valide tôt
    root = Path(output_root) if output_root is not None else base.output.root
    configs = expand_grid(base, grid_file)
    print(f"Grid : {len(configs)} combinaisons")

    for i, cfg in enumerate(configs, 1):
        run_dir = root / dirname_for(cfg)
        if resume and run_dir.exists():
            print(f"[{i}/{len(configs)}] déjà fait, saut : {run_dir.name}")
            continue
        print(f"[{i}/{len(configs)}] {cfg.model_dump(mode='json')}")
        try:
            run_experiment(cfg, output_root=root)
        except Exception as exc:  # noqa: BLE001 — une combo qui casse ne stoppe pas le grid
            traceback.print_exc()
            failed = RunResult(
                config=cfg, task=get_task(cfg.task), histories=[], test_metrics={},
                test_metrics_folds=[], best_state={}, status="failed", error=str(exc),
            )
            append_leaderboard(root / "leaderboard.csv", failed)
```

Note : `dirname_for` inclut un horodatage → `resume` ne peut pas retrouver un dossier par nom exact. Corriger `dirname_for` pour être **déterministe** : `f"{_config_hash(config)}"` seul (retirer l'horodatage), OU ajouter à `RunResult`/`write_run_dir` un test d'existence par hash. **Décision** : `dirname_for` renvoie `_config_hash(config)` seul (8 hex). L'horodatage passe dans `config.yaml` / le leaderboard. Ajuster Task 13 `dirname_for` et son test (`test_dirname_is_deterministic_for_same_config` devient `assert dirname_for(cfg) == dirname_for(cfg)`).

- [ ] **Step 4: Revenir corriger `dirname_for` (Task 13)**

Dans `src/experiments/artifacts.py` :
```python
def dirname_for(config: ExperimentConfig) -> str:
    """Nom de dossier déterministe = hash de la config (8 hex)."""
    return _config_hash(config)
```
Ajuster le test `test_dirname_is_deterministic_for_same_config` :
```python
def test_dirname_is_deterministic_for_same_config():
    assert dirname_for(ExperimentConfig()) == dirname_for(ExperimentConfig())
```
Re-lancer `tests/experiments/test_artifacts.py` → PASS.

- [ ] **Step 5: Vérifier les tests grid**

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/experiments -q`
Expected: PASS

- [ ] **Step 6: Lint + suite + commit**

```bash
ruff check src/experiments tests/experiments
/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q
git add src/experiments/grid.py src/experiments/artifacts.py tests/experiments/test_grid.py tests/experiments/test_artifacts.py
git commit -m "feat: grid search (depliage de grille + execution sequentielle)"
```

---

## Task 16: CLI `run.py` + `grid_cli.py`

**Files:**
- Create: `src/experiments/run.py`, `src/experiments/grid_cli.py`

**Interfaces:**
- Consumes: `src.config.load_config`, `src.experiments.runner.run_experiment`, `src.experiments.grid.run_grid`.
- Produces : deux `main()` exécutables par `python -m src.experiments.run` et `python -m src.experiments.grid`.
  - `run.py` : `--config` (requis), `--set KEY=VAL ...` (`nargs="*"`).
  - `grid_cli.py` : `--config` (requis), `--grid` (requis), `--set ...`, `--task`, `--resume` (`store_true`).

Note : `python -m src.experiments.grid` exécute `src/experiments/grid.py` (qui contient `run_grid`, pas de `main`). Pour éviter la confusion, `grid_cli.py` n'est pas invoqué en `-m src.experiments.grid_cli` mais on ajoute `if __name__ == "__main__": main()` dans **`grid.py`** directement OU on documente `python -m src.experiments.grid_cli`. **Décision** : garder `grid.py` sans `__main__`, la CLI est `python -m src.experiments.grid_cli`. Mettre à jour la spec §10 mentalement (commande `... -m src.experiments.grid_cli`).

- [ ] **Step 1: Implémenter `src/experiments/run.py`**

```python
"""CLI : lance une expérience unique depuis un fichier de config."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Lance une expérience d'entraînement.")
    parser.add_argument("--config", required=True, help="chemin du YAML de config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="CLE=VALEUR",
        help="surcharges ponctuelles, ex. train.nb_epoch=10 loss.name=focal",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    result = run_experiment(config)
    print(f"Run terminé : {result.run_dir}")
    print(f"Métriques test : {result.test_metrics}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implémenter `src/experiments/grid_cli.py`**

```python
"""CLI : lance un grid search (une invocation = une tâche)."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.experiments.grid import run_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Lance un grid search séquentiel.")
    parser.add_argument("--config", required=True, help="config de base (YAML)")
    parser.add_argument("--grid", required=True, help="fichier de grille (YAML)")
    parser.add_argument("--set", nargs="*", default=[], metavar="CLE=VALEUR")
    parser.add_argument("--task", default=None, choices=["dataset", "type", "criticality", "binary_criticality"])
    parser.add_argument("--resume", action="store_true", help="saute les runs déjà présents")
    args = parser.parse_args()
    base = load_config(args.config, args.set)
    run_grid(base, args.grid, task=args.task, resume=args.resume)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke manuel des CLI (help)**

Run:
```
/Users/romain/Projects/Liora/Projet/.venv/bin/python -m src.experiments.run --help
/Users/romain/Projects/Liora/Projet/.venv/bin/python -m src.experiments.grid_cli --help
```
Expected: les deux affichent leur usage sans erreur d'import.

- [ ] **Step 4: Lint + commit**

```bash
ruff check src/experiments/run.py src/experiments/grid_cli.py
git add src/experiments/run.py src/experiments/grid_cli.py
git commit -m "feat: CLI run unique et grid search"
```

---

## Task 17: Configs YAML, suppression de `train.py`, doc

**Files:**
- Create: `configs/base.yaml`, `configs/grids/preprocessing.yaml`, `configs/grids/augmentation.yaml`, `configs/grids/imbalance.yaml`
- Delete: `src/train.py`
- Modify: `README.md` (section usage), `reports/todo.md` (cocher #6/#7 en cours)

- [ ] **Step 1: Créer `configs/base.yaml`**

```yaml
# Config de base — reprend les valeurs par défaut de ExperimentConfig.
# Le grid search tourne par défaut en holdout (validation croisée = coût x N).
task: type
seed: 0

data:
  datasets: [barcelona, munich]
  n_images: 0            # 0 = tout le dataset
  test_size: 0.15
  stratify: type_dataset

transform:
  img_size: [224, 224]
  reinhard: true
  reinhard_samples: 500
  clahe: true
  clahe_clip_limit: 2.0
  clahe_grid_size: 8
  jpeg: true
  jpeg_quality: 35
  p_rotation: 0.9
  p_brightness: 0.5
  p_contrast: 0.5
  brightness_delta: 0.15
  contrast_delta: 0.15

model:
  name: baseline

loss:
  name: cross_entropy
  beta: 0.9999
  gamma: 2.0
  extra_weight: {}
  lambda_cost: 1.0

optim:
  name: adam
  lr: 0.01
  weight_decay: 0.0

scheduler:
  name: none
  patience: 3
  lr_min: 0.0
  step_gamma: 0.1

train:
  nb_epoch: 3
  batch_size: 32
  validation: holdout
  val_size: 0.15
  cross_val_folds: 3
  early_stopping_patience: 0
  device: auto

output:
  root: results
  save_checkpoint: true
  save_figures: true
```

- [ ] **Step 2: Créer les grilles**

`configs/grids/preprocessing.yaml` :
```yaml
transform.reinhard: [true, false]
transform.clahe: [true, false]
transform.jpeg: [true, false]
```

`configs/grids/augmentation.yaml` :
```yaml
transform.p_rotation: [0.0, 0.5, 0.9]
transform.p_brightness: [0.0, 0.5]
transform.p_contrast: [0.0, 0.5]
transform.brightness_delta: [0.1, 0.2]
transform.contrast_delta: [0.1, 0.2]
```

`configs/grids/imbalance.yaml` :
```yaml
- loss.name: [cross_entropy]
- loss.name: [focal]
  loss.beta: [0.99, 0.999, 0.9999]
  loss.gamma: [1.0, 2.0]
```

- [ ] **Step 3: Test de chargement des configs livrées**

Ajouter à `tests/test_config.py` :
```python
def test_shipped_base_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg.task == "type"
    assert cfg.transform.reinhard is True


@pytest.mark.parametrize("grid", ["preprocessing", "augmentation", "imbalance"])
def test_shipped_grids_expand(grid):
    from src.experiments.grid import expand_grid

    configs = expand_grid(load_config("configs/base.yaml"), f"configs/grids/{grid}.yaml")
    assert len(configs) >= 2
```

Run: `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 4: Supprimer `src/train.py`**

```bash
git rm src/train.py
grep -rn "src.train\b\|src/train.py\|from src import train" src tests README.md
```
Expected: aucune occurrence dans `src/` et `tests/`. Corriger `README.md` s'il documente `python -m src.train`.

- [ ] **Step 5: Mettre à jour `README.md`**

Remplacer toute mention de `src/train.py` / `python src/train.py ...` par :
```markdown
## Entraînement

Run unique :
```
uv run python -m src.experiments.run --config configs/base.yaml
uv run python -m src.experiments.run --config configs/base.yaml --set train.nb_epoch=10 loss.name=focal
```

Grid search (une invocation = une tâche) :
```
uv run python -m src.experiments.grid_cli --config configs/base.yaml --grid configs/grids/imbalance.yaml --task type
```

Les résultats sont écrits dans `results/<hash>/` (config figée, history, métriques,
figures, checkpoint) et agrégés dans `results/leaderboard.csv`.
```

- [ ] **Step 6: Suite complète + lint global**

Run: `ruff check src tests && /Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`
Expected: PASS, aucune régression.

- [ ] **Step 7: Vérifier qu'il ne reste aucune référence morte**

Run:
```
grep -rn "base_model\|train_model\|models.callbacks\|src.train " src tests
grep -rn "def run(" src/experiments
```
Expected: 1ère commande → rien ; le pipeline est propre.

- [ ] **Step 8: Commit**

```bash
git add configs README.md reports/todo.md tests/test_config.py
git rm src/train.py
git commit -m "feat: configs YAML livrees + suppression de src/train.py"
```

- [ ] **Step 9: Run manuel de bout en bout** (si données disponibles)

```
uv run python -m src.experiments.run --config configs/base.yaml --set data.n_images=120 train.nb_epoch=1 transform.reinhard=false transform.clahe=false
uv run python -m src.experiments.grid_cli --config configs/base.yaml --grid configs/grids/imbalance.yaml --task type --set data.n_images=120 train.nb_epoch=1 transform.reinhard=false transform.clahe=false
```
Expected: `results/leaderboard.csv` contient les lignes, un dossier par run avec figures + checkpoint.

---

## Self-Review

**1. Spec coverage :**

| Spec § | Tâche(s) |
|---|---|
| §2 carte des modules | Tasks 1-17 (toute la structure) |
| §3 schéma pydantic + `load_config` + overrides | Task 1 |
| §4 `TaskSpec` + `get_task` + warning classes manquantes | Task 2 (spec) + Task 14 (`_warn_if_missing_classes`) |
| §5 `dataframe.py` / `dataset.py` slim / `stats.py` / `transform.py` | Tasks 3, 4, 5, 6 |
| §6 `Trainer` + `components.py` (device, splits, TransformBuilder, optim, scheduler) | Tasks 11, 12 |
| §7 `runner.py` + `RunResult` + comportement par mode de validation | Task 14 |
| §8 grid + `_apply_overrides` + grilles YAML | Tasks 15, 17 |
| §9 artefacts (dossier + leaderboard + hash) | Task 13 (+ correction `dirname_for` Task 15) |
| §10 CLI | Task 16 |
| §11 tests (existants redirigés + neufs) | Tasks 2-16 (chacune ses tests) + Task 10 (callbacks) + Task 12 (trainer) |
| §12 séquence de build | ordre des tâches 1→17 |
| §13 hors périmètre (sampler, transfer, tracking, notebooks) | Task 7 (`NotImplementedError` transfer) ; sampler/tracking non implémentés (conforme) |
| §14 risques (portage boucle, coût Reinhard/CV, tâche `dataset` BCE) | Task 12 (assertions scheduler/ES), Task 8/9 (chemin `dataset`) |

Écart assumé vs spec : la CLI grid est `python -m src.experiments.grid_cli` (et non `... .grid`) pour ne pas mêler module logique et point d'entrée — noté en Task 16. Les notebooks cassés (§13) : la liste des imports à corriger n'est pas produite par une tâche dédiée ; `grep -rn "src.models.train_model\|src.features.dataset import create_dataframe" notebooks` est à lancer par l'utilisateur (hors périmètre confirmé).

**2. Placeholder scan :** aucun "TBD"/"TODO"/"à compléter" dans les steps. Toutes les étapes de code portent le code réel. Les notes "voir Task N" sont accompagnées de l'action concrète (ex. Task 13 → correction `dirname_for` explicitée en Task 15 Step 4).

**3. Type consistency :**
- `build_metrics(task, for_test=False, device=...)` — signature identique Tasks 9, 13, 14. Pas d'incohérence.
- `TransformBuilder(transform_cfg, task, reinhard_normalizer=None)` + `.build(fit_df, *, training)` — cohérent Tasks 5, 11, 14.
- `build_loss(loss_cfg, task, train_df)` — cohérent Tasks 8, 14.
- `RunResult` : champs définis Task 13, `test_metrics_raw` ajouté explicitement Task 13 (note) et posé Task 14. Cohérent.
- `make_splits(train_df, train_cfg, data_cfg)` — signature identique Tasks 11, 14. (La spec §6 disait `make_splits(train_df, train_cfg, task)` ; corrigé ici en `data_cfg` car c'est `data_cfg.stratify` qui pilote la stratification, pas la tâche — écart cohérent et intentionnel.)
- `split_off_test(df, data_cfg)` — la spec §7 le montrait avec `task` en 3e arg ; retiré ici (non utilisé). Cohérent Tasks 11, 14.
- `dirname_for` : rendu déterministe (hash seul) — Task 13 corrigée par Task 15 Step 4, test ajusté au même endroit.
- `EarlyStopping.step(epoch_metrics, model) -> bool` / `.restore(model)` — API inchangée, Tasks 10, 12.

Incohérences corrigées inline : signatures `make_splits` / `split_off_test` alignées sur `data_cfg` ; `dirname_for` déterministe ; CLI grid renommée `grid_cli`.

---

## Execution Handoff

Plan complet et sauvegardé dans `docs/superpowers/plans/2026-09-02-refonte-pipeline-entrainement.md`. Deux options d'exécution :

**1. Subagent-Driven (recommandé)** — je dispatche un subagent neuf par tâche, revue entre chaque tâche, itération rapide.

**2. Inline Execution** — j'exécute les tâches dans cette session via executing-plans, exécution par lots avec points de contrôle.

Quelle approche ?
