"""CLI : inférence d'un modèle entraîné sur une ou plusieurs images."""

from __future__ import annotations

import argparse

from src.experiments.inference import run_inference


def main() -> None:
    """Prédit la classe d'images à partir d'un run existant."""
    parser = argparse.ArgumentParser(description="Inférence depuis un run entraîné.")
    parser.add_argument("--run", required=True, help="dossier du run (config.yaml + checkpoint.pt)")
    parser.add_argument("--image", nargs="+", required=True, help="chemin(s) d'image(s)")
    parser.add_argument("--out", default=None, help="dossier où écrire predictions.json (optionnel)")
    args = parser.parse_args()

    predictions = run_inference(args.run, args.image, output=args.out)
    for pred in predictions:
        top = pred["pred_class"]
        print(f"{pred['path']} -> {top} ({pred['probs'][top]:.3f})")


if __name__ == "__main__":
    main()
