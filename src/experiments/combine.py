"""Combine un run "type" et un run "criticité" : fusion des probas (product of experts).

Les deux modèles sont entraînés indépendamment (voir ``inference.py`` pour le
chargement d'un run isolé) ; ce module ne fait que les charger tous les deux,
vérifier qu'ils forment une paire cohérente, et fusionner leurs probabilités
``P(type=t) = P_type(t) * P_groupe(groupe(t))`` (renormalisé).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.constants import (
    BINARY_CRITICALITY,
    BINARY_CRITICALITY_TO_LABEL,
    CRITICALITY,
    CRITICALITY_TO_LABEL,
    TYPE_TO_LABEL,
)
from src.experiments.inference import load_run, predict_images, rebuild_eval_transform
from src.experiments.runner import seed_everything
from src.features.dataframe import create_dataframe
from src.models.metrics import build_metrics
from src.training.components import split_off_test

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.config import ExperimentConfig
    from src.tasks import TaskSpec

CRITICALITY_TASKS = ("criticality", "binary_criticality")


@dataclass
class LoadedRun:
    """Un run rechargé, prêt à l'emploi (voir ``inference.load_run``)."""

    config: ExperimentConfig
    model: torch.nn.Module
    task: TaskSpec
    run_dir: Path


def load_pair(
    run_a: str | Path, run_b: str | Path, device: str | torch.device = "cpu"
) -> tuple[LoadedRun, LoadedRun]:
    """Charge deux runs et vérifie qu'ils forment une paire ``type`` + criticité valide.

    Args:
        run_a: Dossier du premier run.
        run_b: Dossier du second run.
        device: Device cible des modèles.

    Returns:
        ``(type_run, crit_run)``, peu importe l'ordre passé en entrée.

    Raises:
        ValueError: Si les deux runs ne forment pas exactement une paire
            ``"type"`` + (``"criticality"`` ou ``"binary_criticality"``).
    """
    runs = [
        LoadedRun(*load_run(run_dir, device=device), run_dir=Path(run_dir))
        for run_dir in (run_a, run_b)
    ]
    names = {r.task.name for r in runs}
    if not (
        "type" in names
        and len(names) == 2
        and names - {"type"} <= set(CRITICALITY_TASKS)
    ):
        raise ValueError(
            "load_pair attend une paire type + criticalité "
            f"(criticality/binary_criticality), reçu : {sorted(r.task.name for r in runs)}"
        )
    type_run = next(r for r in runs if r.task.name == "type")
    crit_run = next(r for r in runs if r.task.name != "type")
    return type_run, crit_run


def predict_pair(
    type_run: LoadedRun,
    crit_run: LoadedRun,
    image_paths: Sequence[str | Path],
    device: str | torch.device = "cpu",
) -> list[dict]:
    """Prédit chaque image avec les deux modèles et fusionne leurs probabilités.

    Args:
        type_run: Run ``"type"`` chargé (voir ``load_pair``).
        crit_run: Run ``"criticality"``/``"binary_criticality"`` chargé.
        image_paths: Chemins des images à prédire.
        device: Device cible des modèles.

    Returns:
        Une liste de dicts ``{"path", "pred_index", "pred_class", "probs",
        "type", "criticality"}`` -- ``"type"``/``"criticality"`` sont les
        résultats bruts de chaque modèle pris isolément (mêmes clés que
        ``inference.predict_images``).
    """
    type_transform = rebuild_eval_transform(type_run.config, type_run.run_dir)
    crit_transform = rebuild_eval_transform(crit_run.config, crit_run.run_dir)
    type_preds = predict_images(
        type_run.model, type_transform, type_run.task, image_paths, device=device
    )
    crit_preds = predict_images(
        crit_run.model, crit_transform, crit_run.task, image_paths, device=device
    )
    mapping = type_to_group_mapping(crit_run.task.name)

    results = []
    for type_pred, crit_pred in zip(type_preds, crit_preds):
        type_probs = torch.tensor(
            [type_pred["probs"][name] for name in type_run.task.class_names]
        )
        crit_probs = torch.tensor(
            [crit_pred["probs"][name] for name in crit_run.task.class_names]
        )
        combined = combine_probs(type_probs, crit_probs, mapping)
        index = int(combined.argmax())
        results.append(
            {
                "path": type_pred["path"],
                "pred_index": index,
                "pred_class": type_run.task.class_names[index],
                "probs": {
                    name: combined[i].item() for i, name in enumerate(type_run.task.class_names)
                },
                "type": type_pred,
                "criticality": crit_pred,
            }
        )
    return results


def evaluate_pair(
    type_run: LoadedRun, crit_run: LoadedRun, device: str | torch.device = "cpu"
) -> dict:
    """Évalue la combinaison des deux modèles sur le jeu de test (métriques ``type``).

    Les deux runs doivent partager la même config ``data`` et le même ``seed``
    -- ce sont les deux seuls éléments dont dépend ``split_off_test``
    (``src/training/components.py``), donc la seule façon de garantir que les
    deux modèles sont évalués sur le même split de test.

    Args:
        type_run: Run ``"type"`` chargé (voir ``load_pair``).
        crit_run: Run ``"criticality"``/``"binary_criticality"`` chargé.
        device: Device cible des modèles.

    Returns:
        Le dict de ``build_metrics(type_run.task, for_test=True).compute()``
        calculé sur les probabilités combinées -- passable tel quel à
        ``src.models.metrics.display_metrics``.

    Raises:
        ValueError: Si les configs ``data`` ou les ``seed`` des deux runs diffèrent.
    """
    if type_run.config.data != crit_run.config.data:
        raise ValueError(
            "les deux runs n'ont pas la même config `data` : le split de test "
            "ne serait pas garanti identique entre les deux modèles."
        )
    if type_run.config.seed != crit_run.config.seed:
        raise ValueError(
            "les deux runs n'ont pas le même `seed` : le split de test "
            "ne serait pas garanti identique entre les deux modèles."
        )

    seed_everything(type_run.config.seed)
    df = create_dataframe(type_run.config.data, type_run.task)
    _, test_df = split_off_test(df, type_run.config.data, seed=type_run.config.seed)

    results = predict_pair(type_run, crit_run, test_df.path.tolist(), device=device)
    combined_probs = torch.tensor(
        [[r["probs"][name] for name in type_run.task.class_names] for r in results]
    )
    true_labels = torch.tensor(
        [TYPE_TO_LABEL[t] for t in test_df["type"]], dtype=torch.long
    )

    metrics = build_metrics(type_run.task, for_test=True, device=device)
    metrics.update(combined_probs, true_labels)
    return metrics.compute()


def type_to_group_mapping(crit_task_name: str) -> dict[int, int]:
    """``{index de type -> index de groupe de criticité}`` pour la tâche donnée.

    Args:
        crit_task_name: ``"criticality"`` (3 groupes) ou ``"binary_criticality"``
            (2 groupes).
    """
    if crit_task_name == "criticality":
        names, to_label = CRITICALITY, CRITICALITY_TO_LABEL
    elif crit_task_name == "binary_criticality":
        names, to_label = BINARY_CRITICALITY, BINARY_CRITICALITY_TO_LABEL
    else:
        raise ValueError(f"tâche de criticité inconnue : {crit_task_name!r}")
    return {TYPE_TO_LABEL[type_]: to_label[group] for type_, group in names.items()}


def combine_probs(
    type_probs: torch.Tensor, crit_probs: torch.Tensor, mapping: dict[int, int]
) -> torch.Tensor:
    """Fusionne les probas type/criticité : ``P_type(t) * P_crit(groupe(t))``, renormalisé.

    Args:
        type_probs: Probas softmax sur les types, shape ``(num_types,)``.
        crit_probs: Probas softmax sur les groupes de criticité, shape ``(num_groups,)``.
        mapping: ``{index de type -> index de groupe}`` (voir ``type_to_group_mapping``).

    Returns:
        Les probas type combinées, shape ``(num_types,)``, sommant à 1.
    """
    weights = torch.tensor(
        [crit_probs[mapping[t]] for t in range(len(mapping))], dtype=type_probs.dtype
    )
    combined = type_probs * weights
    return combined / combined.sum()
