"""CLI pour réentraîner depuis zéro les meilleurs modèles."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.retrain import retrain


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sélectionne les meilleurs runs existants, "
            "modifie leur configuration et réentraîne "
            "un nouveau modèle depuis zéro."
        )
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Dossier contenant leaderboard.csv. Défaut : results",
    )

    parser.add_argument(
        "--from-epochs",
        type=int,
        required=True,
        help="Nombre d'epochs des modèles sources.",
    )

    parser.add_argument(
        "--retrain",
        type=Path,
        required=True,
        help="Fichier YAML contenant les paramètres à modifier.",
    )

    parser.add_argument(
        "--top",
        type=float,
        required=True,
        help="Pourcentage des meilleurs modèles à conserver.",
    )

    parser.add_argument(
        "--metric",
        default="val_f1_macro",
        help="Métrique de sélection. Défaut : val_f1_macro",
    )

    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimise la métrique au lieu de la maximiser.",
    )

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="CLE=VALEUR",
        help=(
            "Surcharge de configuration. "
            "Peut être répété plusieurs fois."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ignore les runs cibles déjà présents.",
    )

    args = parser.parse_args()

    retrain(
        root=args.results_root,
        retrain_file=args.retrain,
        source_epochs=args.from_epochs,
        additional_top_percent=args.top,
        overrides=args.set,
        metric=args.metric,
        maximize=not args.minimize,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
