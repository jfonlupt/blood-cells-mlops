"""Écriture des artefacts d'un run : dossier identifié par hash + leaderboard cumulatif."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
import yaml

from src.constants import CRITICAL_LABELS
from src.models.helpers import plot_train_metrics
from src.models.metrics import display_metrics

if TYPE_CHECKING:
    from src.config import ExperimentConfig
    from src.experiments.drift_reference import DriftReference
    from src.tasks import TaskSpec


@dataclass
class RunResult:
    """Résultat complet d'un run, prêt à être sérialisé sur disque."""

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
    test_metrics_raw: dict | None = None
    name: str = ""
    preprocessing: dict | None = None
    val_metrics: dict | None = None
    val_metrics_raw: dict | None = None
    n_trainable_params: int | None = None
    drift_reference: DriftReference | None = None
    mlflow_run_id: str | None = None


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
    """Hash SHA-256 (8 caractères) de la config normalisée en JSON trié."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def dirname_for(config: ExperimentConfig) -> str:
    """Nom de dossier d'un run : uniquement le hash de la config (identité stable)."""
    return _config_hash(config)


def run_dir_for(root: Path, config: ExperimentConfig) -> Path:
    """Chemin du dossier d'un run : ``<root>/runs/<hash de config>``.

    Le leaderboard cumulatif reste, lui, à la racine (``<root>/leaderboard.csv``).
    """
    return Path(root) / "runs" / dirname_for(config)


def write_run_dir(result: RunResult, root: Path, verbose: bool = False) -> Path:
    """Écrit config / history / métriques test / figures / checkpoint / meta. Renvoie le dossier."""
    run_dir = run_dir_for(root, result.config)
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    result.run_dir = run_dir

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(result.config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": result.name,
                "timestamp": f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}",
                "status": result.status,
            },
            indent=2,
        )
    )

    if result.preprocessing is not None:
        (run_dir / "preprocessing.json").write_text(json.dumps(result.preprocessing, indent=2))

    if result.drift_reference is not None:
        drift_dir = run_dir / "drift_reference"
        drift_dir.mkdir(parents=True, exist_ok=True)
        result.drift_reference.descriptors.to_parquet(drift_dir / "descriptors.parquet", index=False)
        if result.drift_reference.embeddings is not None:
            result.drift_reference.embeddings.to_parquet(
                drift_dir / "embeddings.parquet", index=False
            )

    hist_payload = result.histories[0] if len(result.histories) == 1 else result.histories
    (run_dir / "history.json").write_text(json.dumps(hist_payload, indent=2))
    (run_dir / "metrics_test.json").write_text(
        json.dumps(result.test_metrics, indent=2, default=str)
    )
    if result.val_metrics:
        (run_dir / "metrics_val.json").write_text(
            json.dumps(result.val_metrics, indent=2, default=str)
        )

    if result.config.output.save_figures:
        is_cv = result.config.train.validation == "cross"
        if not is_cv and result.histories:
            plot_train_metrics(result.histories[0], save_dir=figures)
        if result.test_metrics_raw is not None:
            display_metrics(
                result.test_metrics_raw,
                num_classes=result.task.num_classes,
                title="Test",
                save_dir=figures,
                verbose=verbose,
            )
        if result.val_metrics_raw is not None:
            display_metrics(
                result.val_metrics_raw,
                num_classes=result.task.num_classes,
                title="Val",
                save_dir=figures,
                verbose=verbose,
            )

    return run_dir


def _min_critical_recall(result: RunResult, prefix: str = "test") -> float:
    """Rappel minimum sur les classes critiques (spec §9).

    Pour ``task="type"`` : min du rappel aux indices ``CRITICAL_LABELS`` (espace des
    types). Pour ``task="criticality"`` : rappel de la classe ``"critique"``.
    ``NaN`` pour toute autre tâche ou si le vecteur de rappel est absent.

    ``prefix`` sélectionne la source : ``"test"`` (défaut, run unique) ou
    ``"val"`` (grid search — le test set n'est jamais évalué par combinaison,
    cf. ``reports/rapport_difficultes_pieges.md``).
    """
    raw = (result.val_metrics_raw if prefix == "val" else result.test_metrics_raw) or {}
    recall = raw.get(f"{prefix}_recall_per_class")
    if not isinstance(recall, list):
        return float("nan")
    if result.config.task == "type":
        indices = CRITICAL_LABELS
    elif result.config.task == "criticality":
        names = list(result.task.class_names)
        indices = [names.index("critique")] if "critique" in names else []
    else:
        return float("nan")
    values = [recall[i] for i in indices if i < len(recall)]
    return min(values) if values else float("nan")


def _val_f1_macro(result: RunResult) -> float:
    """``val_f1`` à l'epoch de ``val_loss`` minimale (spec §9). ``NaN`` si absent."""
    if not result.histories:
        return float("nan")
    history = result.histories[0]
    val_loss = history.get("val_loss")
    val_f1 = history.get("val_f1")
    if not val_loss or not val_f1:
        return float("nan")
    best_epoch = min(range(len(val_loss)), key=lambda i: val_loss[i])
    return val_f1[best_epoch] if best_epoch < len(val_f1) else float("nan")


def append_leaderboard(path: Path, result: RunResult) -> None:
    """Ajoute une ligne au CSV cumulatif (entête créé si le fichier est absent)."""
    path = Path(path)
    row = {
        "run_dir": str(result.run_dir) if result.run_dir else "",
        "name": result.name,
        "timestamp": f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S}",
        "task": result.config.task,
        "seed": result.config.seed,
        "duration_s": round(result.duration_s, 1),
        "status": result.status,
        "error": result.error,
        "min_critical_recall": _min_critical_recall(result, prefix="test"),
        "val_min_critical_recall": _min_critical_recall(result, prefix="val"),
        "val_f1_macro": _val_f1_macro(result),
        "n_trainable_params": result.n_trainable_params,
    }
    row.update(flatten(result.config.model_dump(mode="json")))
    # ``loss.name`` a été retiré de la config (perte unique) : on garde la colonne
    # du leaderboard, remplie d'un nom générique, pour ne pas casser l'historique.
    row["loss.name"] = "clinical"
    row.update(
        {k if k.startswith("test_") else f"test_{k}": v for k, v in result.test_metrics.items()}
    )
    # Métriques riches sur validation (f1 macro, mcc, per-class, top3...), toujours
    # disponibles (calculées sur val_ds, jamais sur le test set) — utilisées pour
    # comparer les combinaisons d'un grid search. Écrase "val_f1_macro" ci-dessus
    # par la version fraîchement recalculée si présente (plus riche que celle tirée
    # de l'historique d'entraînement).
    row.update(result.val_metrics or {})
    new_row = pd.DataFrame([row])
    # Si le CSV existe, on le relit et on concatène : pandas aligne les colonnes
    # par nom (NaN pour les manquantes), ce qui évite le décalage positionnel
    # quand des runs successifs ont des colonnes distinctes (holdout vs CV, type
    # vs dataset).
    frame = pd.concat([pd.read_csv(path), new_row], ignore_index=True) if path.exists() else new_row
    frame.to_csv(path, index=False)
