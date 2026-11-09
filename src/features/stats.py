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
    """Moyenne / écart-type par canal, calculés en neutralisant la normalisation finale.

    Args:
        train_df: DataFrame du split d'entraînement (colonnes ``path`` + label).
        transform_cfg: Configuration de prétraitement.
        task: Spécification de la tâche (colonne / type de label).
        reinhard_normalizer: Normaliseur Reinhard déjà fitté, ou ``None`` pour ignorer.
        n_batches: Nombre de batches échantillonnés pour l'estimation.
        batch_size: Taille de batch du ``DataLoader`` sonde.

    Returns:
        Le couple ``(mean, std)`` de tenseurs de forme ``(3,)`` (float32).
    """
    # La sonde doit mesurer les stats *avant* normalisation finale : on force
    # norm_stats='dataset' + une normalisation identité (mean 0, std 1), quel que
    # soit le norm_stats demandé pour l'entraînement.
    probe_cfg = transform_cfg.model_copy(update={"norm_stats": "dataset"})
    probe_transform = build_transform(
        probe_cfg,
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
    """Moyenne les stats Lab cible de Reinhard sur un échantillon (équilibré par origine).

    Plutôt qu'une seule image de référence, moyenne les ``target_means`` /
    ``target_stds`` sur plusieurs images pour viser une cible plus robuste
    (ex. à parts égales entre datasets d'origine).

    Args:
        normalizer: ``ReinhardNormalizer`` (torchstain) à ajuster en place.
        target_image_paths: Chemins des images candidates.
        img_size: Taille ``(H, W)`` de redimensionnement avant fit.
        n_samples: Nombre total d'images à échantillonner.
        origins: Origine de chaque image (même longueur que ``target_image_paths``),
            ou ``None`` pour un échantillonnage uniforme.
        seed: Graine du tirage aléatoire.

    Returns:
        Le ``normalizer`` avec ses stats cible remplacées par les moyennes.
    """
    rng = random.Random(seed)  # noqa: S311 - échantillonnage reproductible, pas cryptographique
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
    """Fitte un ``ReinhardNormalizer`` sur une image de référence unique.

    Args:
        target_image_path: Chemin de l'image de référence.
        img_size: Taille ``(H, W)`` de redimensionnement avant fit.
        method: ``None`` -> Reinhard standard ; ``"modified"`` -> variante torchstain.

    Returns:
        Le ``ReinhardNormalizer`` fitté.
    """
    target = resize_with_pad(load_image_rgb(target_image_path), img_size, fill=255).float()
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend="torch", method=method)
    normalizer.fit(target)
    return normalizer


def build_reinhard(train_df: pd.DataFrame, transform_cfg: TransformConfig, seed: int = 0):
    """Cible Reinhard moyennée sur le train, équilibrée entre datasets d'origine.

    Args:
        train_df: DataFrame du split d'entraînement (colonnes ``path`` + ``dataset``).
        transform_cfg: Configuration de prétraitement (``reinhard_samples``).
        seed: Graine du tirage des images cible pour la reproductibilité.

    Returns:
        Un ``ReinhardNormalizer`` dont les stats cible sont moyennées sur le train.
    """
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend="torch", method=None)
    return average_reinhard_target_stats(
        normalizer,
        list(train_df.path),
        origins=list(train_df.dataset),
        n_samples=transform_cfg.reinhard_samples,
        seed=seed,
    )
