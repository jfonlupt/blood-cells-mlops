"""CLI : lance un grid search (une invocation = une tâche)."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.experiments.grid import run_grid


def main() -> None:
    """Lance un grid search séquentiel depuis des fichiers de config et grille."""
    parser = argparse.ArgumentParser(description="Lance un grid search séquentiel.")
    parser.add_argument("--config", required=True, help="config de base (YAML)")
    parser.add_argument("--grid", required=True, help="fichier de grille (YAML)")
    parser.add_argument("--set", nargs="*", default=[], metavar="CLE=VALEUR")
    parser.add_argument(
        "--task",
        default=None,
        choices=["dataset", "type", "criticality", "binary_criticality"],
    )
    parser.add_argument("--resume", action="store_true", help="saute les runs déjà présents")
    parser.add_argument(
        "--name", default=None,
        help="nom partagé par tous les runs de la grille (défaut : nom aléatoire)",
    )
    args = parser.parse_args()
    base = load_config(args.config, args.set)
    run_grid(base, args.grid, task=args.task, resume=args.resume, name=args.name)


if __name__ == "__main__":
    main()
