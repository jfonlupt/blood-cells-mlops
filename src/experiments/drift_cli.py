"""CLI : rapport de data drift entre le modèle champion et de nouvelles images."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.drift import run_drift_report

# Extensions d'image reconnues (insensible à la casse) : filtre les fichiers
# parasites d'un dossier (.DS_Store, Thumbs.db, etc.) qui sinon font échouer
# compute_descriptors (OSError "image illisible") et avortent tout le rapport.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare de nouvelles images à la référence de drift du modèle champion."
    )
    parser.add_argument("--model", required=True, help="nom du modèle enregistré dans MLflow")
    parser.add_argument("--images", required=True, type=Path, help="dossier d'images à comparer")
    parser.add_argument(
        "--alias", default="champion", help="alias MLflow à résoudre. Défaut : champion"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("drift_report"), help="dossier de sortie du rapport"
    )
    args = parser.parse_args()

    image_paths = sorted(
        p
        for p in args.images.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    result = run_drift_report(args.model, image_paths, alias=args.alias, output=args.output)

    print(f"Rapport écrit dans {result.report_path}")
    print(f"Drift descripteurs : {result.descriptors_drifted}")
    if result.embeddings_drifted is not None:
        print(f"Drift embeddings : {result.embeddings_drifted}")


if __name__ == "__main__":
    main()
