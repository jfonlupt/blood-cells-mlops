"""Grid search : déplie une grille de surcharges en configs et lance chaque run."""

from __future__ import annotations

import random
import traceback
from pathlib import Path

import yaml
from coolname import generate_slug
from sklearn.model_selection import ParameterGrid

from src.config import ExperimentConfig, deep_merge
from src.experiments import tracking
from src.experiments.artifacts import RunResult, append_leaderboard, run_dir_for
from src.experiments.runner import run_experiment
from src.tasks import get_task


def _nest(flat: dict) -> dict:
    """Convertit un dict à clés pointées en dict imbriqué.

    Args:
        flat: Dict dont les clés sont pointées, ex. ``{"a.b": 1}``.

    Returns:
        Le dict imbriqué correspondant, ex. ``{"a": {"b": 1}}``.
    """
    out: dict = {}
    for dotted, value in flat.items():
        node = out
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return out


def _apply_overrides(base: ExperimentConfig, flat: dict) -> ExperimentConfig:
    """Applique des surcharges à clés pointées sur une config de base.

    Args:
        base: Config de référence.
        flat: Surcharges à clés pointées, ex. ``{"loss.gamma": 2.0}``.

    Returns:
        Une nouvelle ``ExperimentConfig`` validée après fusion.
    """
    merged = deep_merge(base.model_dump(mode="json"), _nest(flat))
    return ExperimentConfig.model_validate(merged)


def expand_grid(base: ExperimentConfig, grid_file: Path) -> list[ExperimentConfig]:
    """Déplie un fichier de grille YAML en une liste de configs.

    Args:
        base: Config de référence sur laquelle appliquer chaque combinaison.
        grid_file: Fichier YAML (dict de listes, ou liste de dicts de listes).

    Returns:
        Une config validée par combinaison du produit cartésien.
    """
    raw = yaml.safe_load(Path(grid_file).read_text())
    return [_apply_overrides(base, combo) for combo in ParameterGrid(raw)]


def run_grid(
    base: ExperimentConfig,
    grid_file: Path,
    *,
    task: str | None = None,
    resume: bool = False,
    output_root: Path | None = None,
    name: str | None = None,
) -> None:
    """Déplie une grille et exécute chaque combinaison séquentiellement.

    Args:
        base: Config de référence.
        grid_file: Fichier YAML décrivant la grille de surcharges.
        task: Nom de tâche à forcer sur toutes les combinaisons ; sinon celui de ``base``.
        resume: Si vrai, saute les combinaisons dont le dossier de run existe déjà.
        output_root: Racine des artefacts ; par défaut ``base.output.root``.
        name: Nom partagé par tous les runs de la grille (colonne ``name`` du
            leaderboard). Si ``None``, un nom aléatoire est tiré une seule fois
            pour toute la grille.
    """
    if task:
        base = _apply_overrides(base, {"task": task})
        get_task(task)  # valide tôt
    name = name or generate_slug(2)
    root = Path(output_root) if output_root is not None else base.output.root
    configs = expand_grid(base, grid_file)
    if base.grid.max_combinations is not None and len(configs) > base.grid.max_combinations:
        configs = random.sample(configs, base.grid.max_combinations)
    base.verbose = False
    print(f"Grid : {len(configs)} combinaisons")

    with tracking.grid_run(name, base.task):
        for i, cfg in enumerate(configs, 1):
            run_dir = run_dir_for(root, cfg)
            if resume and run_dir.exists():
                print(f"[{i}/{len(configs)}] déjà fait, saut : {run_dir.name}")
                continue
            print(f"[{i}/{len(configs)}] {cfg.model_dump(mode='json')}")
            try:
                run_experiment(cfg, output_root=root, name=name, evaluate_test=False)
            except Exception as exc:  # noqa: BLE001 -- une combo qui casse ne stoppe pas le grid entier
                traceback.print_exc()
                failed = RunResult(
                    config=cfg,
                    task=get_task(cfg.task),
                    histories=[],
                    test_metrics={},
                    test_metrics_folds=[],
                    best_state={},
                    status="failed",
                    error=str(exc),
                    name=name,
                )
                append_leaderboard(root / "leaderboard.csv", failed)
