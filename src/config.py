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
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    """Sélection et découpe des jeux de données."""

    datasets: list[
        Literal[
            "barcelona",
            "munich",
            "munich_train",
            "munich_eval",
            "data_drift_train",
            "data_drift_eval",
        ]
    ] = ["barcelona", "munich"]
    n_images: int = 0
    test_size: float = 0.15
    stratify: Literal[
        "no", "type", "dataset", "type_dataset", "criticality", "binary_criticality"
    ] = "type_dataset"


class TransformConfig(_Base):
    """Prétraitement et augmentation des images."""

    img_size: tuple[int, int] = (224, 224)
    norm_stats: Literal["dataset", "imagenet"] = "dataset"
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
    p_flip: float = 0.5
    p_blur: float = 0.0
    blur_kernel_size: int = 5
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 2.0


class LossConfig(_Base):
    """Hyperparamètres de la perte unique (cf. ``src/models/losses.py::ClinicalLoss``).

    ``L_i = -w_y (1-p_y)**gamma log p_y + lambda_cost * sum_c M[y,c] p_c``

    - ``beta`` : pondération class-balanced (0 = non pondéré, 1 = inverse-fréquence) ;
    - ``gamma`` : terme focal (0 = désactivé) ;
    - ``lambda_cost`` : poids du terme de coût espéré (0 = désactivé) ;
    - ``extra_weight`` : multiplicateur manuel par classe (boost clinique) ;
    - ``label_smoothing`` : lissage de labels (0 = désactivé) ;
    - ``cost_matrix`` : matrice de coût ``M`` (``None`` = matrice par défaut).
      Pour ``task=type`` une matrice 2x2 ou 3x3 est *liftée* via la criticité des
      classes ; une matrice à la dimension de la tâche est utilisée telle quelle.
    """

    beta: float = Field(0.9999, ge=0, le=1)
    gamma: float = Field(0.0, ge=0)
    lambda_cost: float = Field(0.0, ge=0)
    extra_weight: dict[str, float] = {}
    label_smoothing: float = Field(0.0, ge=0, lt=1)
    cost_matrix: list[list[float]] | None = None

    @field_validator("cost_matrix")
    @classmethod
    def _check_cost_matrix(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        """Vérifie que la matrice est carrée et d'ordre >= 2 (dimension vs tâche : cf. build_loss)."""
        if value is None:
            return value
        n = len(value)
        if n < 2 or any(len(row) != n for row in value):
            raise ValueError(
                f"loss.cost_matrix doit être carrée et d'ordre >= 2 (reçu {n} lignes)."
            )
        return value


class OptimConfig(_Base):
    """Optimiseur."""

    name: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = 1e-2
    weight_decay: float = 0.0


class SchedulerConfig(_Base):
    """Ordonnanceur de taux d'apprentissage."""

    name: Literal["none", "plateau", "cosine", "step"] = "none"
    patience: int = 3
    lr_min: float = 0.0
    step_gamma: float = 0.1


class TrainConfig(_Base):
    """Paramètres de la boucle d'entraînement."""

    nb_epoch: int = 3
    batch_size: int = 32
    validation: Literal["none", "holdout", "cross"] = "holdout"
    val_size: float = 0.15
    cross_val_folds: int = 3
    early_stopping_patience: int = 0
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    sampler: Literal["none", "weight"] = "none"
    sampler_alpha: float = Field(1.0, ge=0)
    num_workers: int = Field(0, ge=0)
    grad_clip_norm: float = Field(1.0, ge=0)


class OutputConfig(_Base):
    """Emplacement et nature des artefacts produits."""

    root: Path = Path("results")
    save_checkpoint: bool = True
    save_figures: bool = True


class GridSearchConfig(_Base):
    """Configuration d'une recherche par grille."""

    max_combinations: int | None = None


class BlockConfig(BaseModel):
    """Un bloc convolutif de ``MultiArchiModel`` (profondeur, canaux, noyau)."""

    depth: int
    channels: int
    kernel: int


class ArchitectureConfig(_Base):
    """Configuration d'une architecture multi-blocs."""

    blocks: list[BlockConfig]
    classifier: list[int]
    use_batchnorm: bool = True
    use_gap: bool = True


class TransferConfig(_Base):
    """Backbone pré-entraîné (timm) + tête de classification + stratégie de fine-tuning."""

    backbone: str = "resnet50"
    pretrained: bool = True
    freeze: Literal["none", "all"] | StrictInt = "none"
    head_hidden: list[int] = []
    head_dropout: float = Field(0.0, ge=0, lt=1)
    backbone_lr_mult: float = Field(1.0, ge=0)

    @model_validator(mode="after")
    def _check_freeze(self) -> TransferConfig:
        """``freeze`` entier = nombre de groupes de stages laissés entraînables (>= 1)."""
        if isinstance(self.freeze, int) and self.freeze < 1:
            raise ValueError(
                "model.transfer.freeze : 'none', 'all', ou un entier >= 1 "
                "(nombre de groupes de stages du backbone laissés entraînables)."
            )
        return self


class ModelConfig(_Base):
    """Choix du modèle."""

    name: Literal["baseline", "dataset", "transfer", "multi_archi"] = "baseline"
    architecture: ArchitectureConfig | None = None
    transfer: TransferConfig | None = None

    @model_validator(mode="after")
    def _check_submodel_present(self) -> ModelConfig:
        """Vérifie que la sous-config exigée par ``name`` est renseignée."""
        if self.name == "transfer" and self.transfer is None:
            raise ValueError("model.name='transfer' exige un bloc model.transfer.")
        if self.name == "multi_archi" and self.architecture is None:
            raise ValueError("model.name='multi_archi' exige un bloc model.architecture.")
        return self


class ExperimentConfig(_Base):
    """Racine de la configuration d'une expérience."""

    task: Literal["dataset", "type", "criticality", "binary_criticality"] = "type"
    verbose: bool = True
    seed: int = 0
    grid: GridSearchConfig = GridSearchConfig()
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
        """Vérifie qu'un jeu de validation existe si plateau ou early stopping."""
        needs_val = self.train.early_stopping_patience > 0 or self.scheduler.name == "plateau"
        if needs_val and self.train.validation == "none":
            raise ValueError(
                "scheduler='plateau' et early_stopping_patience>0 exigent "
                "train.validation='holdout' ou 'cross'."
            )
        return self


def _coerce(value: str) -> Any:
    """Convertit une valeur d'override en littéral Python, sinon la laisse en str."""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null":
        return None
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
