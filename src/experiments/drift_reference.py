"""Construction de la référence de drift (échantillon + descripteurs + embeddings) d'un run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import torch

from src.features.drift_descriptors import compute_descriptors
from src.features.transform import load_image_rgb

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

# Dimension cible de l'embedding de drift, par moyenne de blocs contigus (voir _pool_embedding) --
# n'affecte jamais le modèle lui-même, seulement la référence stockée pour le drift. Certains
# modèles (ex. BaselineClassifier, dont .features() n'a pas de pooling global avant le flatten)
# renvoient un vecteur brut de dizaines de milliers de dimensions ; sans réduction, leur
# embeddings.parquet de référence pèse ~90 Mo pour un modèle de 11 Mo. 128 est un compromis
# taille/signal : les modèles déjà compacts (ex. TransferModel, backbone pré-entraîné avec
# pooling global) ne sont pas affectés (no-op si dim <= 128).
_EMBEDDING_TARGET_DIM = 128


def _pool_embedding(features: torch.Tensor, target_dim: int = _EMBEDDING_TARGET_DIM) -> torch.Tensor:
    """Réduit ``features`` (B, D) à (B, min(D, target_dim)) par moyenne de blocs contigus.

    No-op si ``D <= target_dim``. ``torch.tensor_split`` découpe en exactement
    ``target_dim`` blocs de tailles quasi égales (à 1 près) même quand ``D`` n'est pas un
    multiple de ``target_dim`` -- déterministe et sans état à sauvegarder : appliqué
    identiquement à la référence (entraînement) et au courant (évaluation), tant que
    l'architecture du modèle (donc ``D``) ne change pas entre les deux.

    C'est une compression grossière (pas une PCA) : elle perd la structure fine de
    l'embedding d'origine, acceptable pour détecter un drift statistique mais pas pour un
    usage nécessitant la représentation exacte du modèle.
    """
    dim = features.shape[1]
    if dim <= target_dim:
        return features
    chunks = torch.tensor_split(features, target_dim, dim=1)
    return torch.stack([c.mean(dim=1) for c in chunks], dim=1)


def sample_stratified(
    df: pd.DataFrame, label_column: str, n_per_class: int, seed: int
) -> pd.DataFrame:
    """Échantillon équilibré par classe : jusqu'à ``n_per_class`` lignes par valeur de ``label_column``.

    Une classe comptant moins de ``n_per_class`` lignes est entièrement conservée
    (pas d'erreur, pas de sur-échantillonnage).

    Args:
        df: DataFrame source (typiquement ``train_df``).
        label_column: Colonne de classe (``task.label_column``).
        n_per_class: Nombre maximal de lignes tirées par classe.
        seed: Graine du tirage, pour la reproductibilité.

    Returns:
        Le sous-DataFrame concaténé, dans l'ordre des classes trié.
    """
    parts = [
        group.sample(n=min(n_per_class, len(group)), random_state=seed)
        for _, group in df.groupby(label_column, sort=True)
    ]
    return pd.concat(parts, ignore_index=True)


@dataclass
class DriftReference:
    """Référence de drift d'un run : descripteurs visuels + embeddings optionnels."""

    descriptors: pd.DataFrame
    embeddings: pd.DataFrame | None


def build_drift_reference(
    paths: list[str],
    eval_transform: Callable,
    model: nn.Module,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
) -> DriftReference:
    """Descripteurs (toujours) + embeddings (si ``model`` expose ``.features()``).

    Args:
        sample_df: List des chemins des images courantes.
        eval_transform: Transform d'évaluation complet (celui utilisé en inférence).
        model: Modèle dont on extrait les features ; ignoré si pas de ``.features()``.
        device: Device sur lequel exécuter ``model.features``.
        batch_size: Taille de batch pour éviter de charger tout l'échantillon en mémoire.

    Returns:
        La ``DriftReference`` ; ``embeddings`` est ``None`` si ``model`` n'a pas ``.features()``.
    """
    # Normalisation en str indispensable : compute_descriptors() construit sa colonne "path"
    # via str(path), donc si l'appelant passe des pathlib.Path (ex. server.py /evaluate, qui
    # reçoit des chemins temporaires BentoML en Path) sans cette normalisation, la colonne
    # "path" de embeddings (Path, insérée brute plus bas) ne correspond plus à celle de
    # descriptors (str) : le merge("path") en aval ne matche alors plus aucune ligne et
    # produit un DataFrame vide, d'où le "empty column" levé par evidently.
    paths = [str(p) for p in paths]
    descriptors = compute_descriptors(paths)
    if not hasattr(model, "features"):
        return DriftReference(descriptors=descriptors, embeddings=None)

    was_training = model.training
    model.eval()
    try:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                batch = torch.stack([eval_transform(load_image_rgb(p)) for p in batch_paths]).to(
                    device
                )
                chunks.append(_pool_embedding(model.features(batch)).cpu())
        features = torch.cat(chunks, dim=0).numpy()
        embeddings = pd.DataFrame(features, columns=[f"emb_{i}" for i in range(features.shape[1])])
        embeddings.insert(0, "path", paths)
        return DriftReference(descriptors=descriptors, embeddings=embeddings)
    finally:
        model.train(was_training)
