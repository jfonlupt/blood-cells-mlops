"""CLI : lance une expérience unique depuis un fichier de config."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.experiments.runner import run_experiment


def main() -> None:
    """Lance une expérience d'entraînement depuis un fichier de config."""
    parser = argparse.ArgumentParser(description="Lance une expérience d'entraînement.")
    parser.add_argument("--config", required=True, help="chemin du YAML de config")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="CLE=VALEUR",
        help="surcharges ponctuelles, ex. train.nb_epoch=10 loss.gamma=2.0",
    )
    parser.add_argument(
        "--name", default=None,
        help="nom lisible du run (défaut : nom aléatoire type 'brave-otter')",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    result = run_experiment(config, name=args.name)
    print(f"Run terminé : {result.run_dir}")
    print(f"Métriques test : {result.test_metrics}")
    print(f"MLflow run id : {result.mlflow_run_id}")


if __name__ == "__main__":
    main()
