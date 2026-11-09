"""Composants assemblés par le runner : device, splits, transforms, optim, scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import WeightedRandomSampler

from src.features.stats import compute_dataset_mean_std
from src.features.transform import IMAGENET_MEAN, IMAGENET_STD, build_transform

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import pandas as pd
    from torchvision.transforms import v2

    from src.config import DataConfig, OptimConfig, SchedulerConfig, TrainConfig, TransformConfig
    from src.tasks import TaskSpec


def resolve_device(name: str) -> torch.device:
    """Résout le device demandé.

    Args:
        name: ``"auto"`` pour choisir cuda / mps / cpu selon disponibilité, sinon un nom explicite.

    Returns:
        Le ``torch.device`` correspondant.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _strat_key(df: pd.DataFrame, data_cfg: DataConfig):
    """Renvoie la colonne de stratification, ou ``None`` si ``stratify == "no"``."""
    return None if data_cfg.stratify == "no" else df[data_cfg.stratify]


def split_off_test(
    df: pd.DataFrame, data_cfg: DataConfig, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare un jeu de test stratifié.

    Args:
        df: DataFrame complet.
        data_cfg: Configuration des données (``test_size``, ``stratify``).
        seed: Graine passée en ``random_state`` pour un découpage reproductible.

    Returns:
        Le couple ``(train, test)``.
    """
    return train_test_split(
        df, test_size=data_cfg.test_size, stratify=_strat_key(df, data_cfg), random_state=seed
    )


def make_splits(
    train_df: pd.DataFrame, train_cfg: TrainConfig, data_cfg: DataConfig, seed: int | None = None
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame | None]]:
    """Itère les paires ``(train, val)`` selon le mode de validation.

    Args:
        train_df: DataFrame d'entraînement (hors test).
        train_cfg: Configuration d'entraînement (``validation``, ``val_size``, ``cross_val_folds``).
        data_cfg: Configuration des données (``stratify``).
        seed: Graine passée en ``random_state`` des splitters pour la reproductibilité.

    Yields:
        ``none`` -> une paire ``(df, None)`` ; ``holdout`` -> une paire ``(tr, va)`` stratifiée ;
        ``cross`` -> ``cross_val_folds`` paires via ``StratifiedKFold`` (ou ``KFold`` si pas de
        stratification).
    """
    if train_cfg.validation == "none":
        yield train_df, None
        return
    if train_cfg.validation == "holdout":
        tr, va = train_test_split(
            train_df,
            test_size=train_cfg.val_size,
            stratify=_strat_key(train_df, data_cfg),
            random_state=seed,
        )
        yield tr, va
        return
    # cross
    if data_cfg.stratify == "no":
        splitter = KFold(n_splits=train_cfg.cross_val_folds, shuffle=True, random_state=seed)
        indices = splitter.split(train_df)
    else:
        splitter = StratifiedKFold(
            n_splits=train_cfg.cross_val_folds, shuffle=True, random_state=seed
        )
        indices = splitter.split(train_df, train_df[data_cfg.stratify])
    for train_idx, val_idx in indices:
        yield train_df.iloc[train_idx], train_df.iloc[val_idx]


class TransformBuilder:
    """Fabrique de transforms : calcule mean/std sur le sous-ensemble d'entraînement fourni."""

    def __init__(self, transform_cfg: TransformConfig, task: TaskSpec, reinhard_normalizer=None):
        """Initialise la fabrique.

        Args:
            transform_cfg: Configuration de prétraitement.
            task: Spécification de la tâche (nécessaire à ``compute_dataset_mean_std``).
            reinhard_normalizer: Normaliseur Reinhard déjà fitté, ou ``None``.
        """
        self.cfg = transform_cfg
        self.task = task
        self.reinhard_normalizer = reinhard_normalizer
        self.dataset_mean: torch.Tensor | None = None
        self.dataset_std: torch.Tensor | None = None

    def build(self, fit_df: pd.DataFrame, *, training: bool) -> v2.Compose:
        """Construit le ``v2.Compose`` pour un split.

        Args:
            fit_df: DataFrame servant à estimer la moyenne / l'écart-type.
            training: Si vrai, ajoute les étapes d'augmentation aléatoire.

        Returns:
            Le ``v2.Compose`` prêt à l'emploi.

        Effet de bord : ``self.dataset_mean`` / ``self.dataset_std`` conservent
        les dernières statistiques estimées (réutilisées pour persister le
        prétraitement du run). Avec ``norm_stats="imagenet"``, ce sont les
        constantes ImageNet et la sonde ``compute_dataset_mean_std`` est évitée.
        """
        if self.cfg.norm_stats == "imagenet":
            mean = torch.tensor(IMAGENET_MEAN)
            std = torch.tensor(IMAGENET_STD)
        else:
            mean, std = compute_dataset_mean_std(
                fit_df, self.cfg, self.task, reinhard_normalizer=self.reinhard_normalizer
            )
        self.dataset_mean, self.dataset_std = mean, std
        return build_transform(
            self.cfg,
            training=training,
            reinhard_normalizer=self.reinhard_normalizer,
            dataset_mean=mean,
            dataset_std=std,
        )


def optimizer_param_groups(model, base_lr: float):
    """Paramètres à passer à l'optimiseur pour ``model``.

    Si ``model`` expose ``param_groups(base_lr)`` (cas de ``TransferModel``, qui
    applique un LR réduit au backbone pré-entraîné), on l'utilise ; sinon on
    renvoie simplement ``model.parameters()``.

    Args:
        model: Le modèle instancié.
        base_lr: Taux d'apprentissage de référence (``optim.lr``).

    Returns:
        Une liste de groupes de paramètres, ou l'itérateur ``model.parameters()``.
    """
    if hasattr(model, "param_groups"):
        return model.param_groups(base_lr)
    return model.parameters()


def build_optimizer(optim_cfg: OptimConfig, params) -> torch.optim.Optimizer:
    """Construit l'optimiseur (Adam ou SGD).

    Args:
        optim_cfg: Configuration (``name``, ``lr``, ``weight_decay``).
        params: Paramètres à optimiser.

    Returns:
        L'optimiseur instancié.
    """
    if optim_cfg.name == "adam":
        return Adam(params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    if optim_cfg.name == "adamw":
        return AdamW(params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    if optim_cfg.name == "sgd":
        return SGD(params, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    raise ValueError(f"optimizer inconnu : {optim_cfg.name!r}")


def build_scheduler(sched_cfg: SchedulerConfig, optimizer, *, nb_epoch: int):
    """Construit le scheduler de learning rate, ou ``None``.

    Args:
        sched_cfg: Configuration (``name``, ``patience``, ``lr_min``, ``step_gamma``).
        optimizer: Optimiseur à piloter.
        nb_epoch: Nombre total d'epochs (pour ``cosine`` et ``step``).

    Returns:
        Le scheduler instancié, ou ``None`` si ``name == "none"``.
    """
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


def build_sampler(
    train_cfg: TrainConfig, labels: Sequence[int] | np.ndarray, num_classes: int
) -> WeightedRandomSampler | None:
    """Construit un ``WeightedRandomSampler`` pour rééquilibrer les classes, ou ``None``.

    Le poids d'une classe est ``(1 / effectif) ** train_cfg.sampler_alpha``, puis
    normalisé pour que la moyenne des poids de classe vaille 1 (même convention que
    ``effective_number_weights`` côté loss). ``sampler_alpha`` règle l'intensité :
    0 = tirage uniforme (le sampler ne change rien), 1 = inverse-fréquence pleine,
    0.5 = racine. Une valeur intermédiaire permet de partager la correction du
    déséquilibre avec une loss pondérée sans la compter deux fois.

    Args:
        train_cfg: Configuration d'entraînement (``sampler``, ``sampler_alpha``).
        labels: Labels entiers du split d'entraînement, dans l'ordre du dataset.
        num_classes: Nombre de classes de la tâche.

    Returns:
        Le ``WeightedRandomSampler`` (tirage avec remise, ``num_samples`` = taille du
        split), ou ``None`` si ``train_cfg.sampler != "weight"``.
    """
    if train_cfg.sampler != "weight":
        return None

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # classe absente du split : évite la division par zéro

    class_weights = (1.0 / counts) ** train_cfg.sampler_alpha
    class_weights = class_weights / class_weights.sum() * num_classes

    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
