"""CLI : combine un run "type" et un run "criticité" (prédiction ou évaluation test)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments.combine import evaluate_pair, load_pair, predict_pair
from src.models.metrics import display_metrics


def main() -> None:
    """Combine deux runs (type + criticité), en prédiction ou en évaluation test."""
    parser = argparse.ArgumentParser(
        description="Combine un run 'type' et un run 'criticality'/'binary_criticality' "
        "(fusion des probabilités, cf. reports/todo.md)."
    )
    parser.add_argument("--run-a", required=True, help="dossier du premier run")
    parser.add_argument("--run-b", required=True, help="dossier du second run")
    parser.add_argument("--image", nargs="+", default=None, help="chemin(s) d'image(s) à prédire")
    parser.add_argument(
        "--evaluate", action="store_true", help="évalue la combinaison sur le jeu de test"
    )
    parser.add_argument("--out", default=None, help="dossier où écrire le résultat (optionnel)")
    args = parser.parse_args()

    if bool(args.image) == bool(args.evaluate):
        parser.error("préciser exactement un des deux : --image ... ou --evaluate")

    type_run, crit_run = load_pair(args.run_a, args.run_b)

    if args.evaluate:
        metrics = evaluate_pair(type_run, crit_run)
        display_metrics(metrics, num_classes=type_run.task.num_classes, title="Test combiné")
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics_combined.json").write_text(
                json.dumps(dict(metrics), default=str, indent=2)
            )
        return

    predictions = predict_pair(type_run, crit_run, args.image)
    for pred in predictions:
        print(f"{pred['path']} -> {pred['pred_class']} ({pred['probs'][pred['pred_class']]:.3f})")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "predictions_combined.json").write_text(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
