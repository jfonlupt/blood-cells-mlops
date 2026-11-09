"""CLI de promotion des meilleurs modèles."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.promote import promote


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sélectionne et réentraîne les meilleurs runs."
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
        "--top",
        type=float,
        required=True,
        help="Pourcentage des modèles à conserver.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Nombre d'epochs supplémentaires.",
    )

    parser.add_argument(
        "--metric",
        default="test_f1_macro",
        help="Métrique de sélection. Défaut : val_f1_macro",
    )

    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Minimise la métrique au lieu de la maximiser.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ignore les promotions dont le dossier cible existe déjà.",
    )

    args = parser.parse_args()

    promote(
        root=args.results_root,
        source_epochs=args.from_epochs,
        additional_epochs=args.epochs,
        top_percent=args.top,
        metric=args.metric,
        maximize=not args.minimize,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
