"""CLI : Grad-CAM d'un modèle entraîné sur une image.

Sortie : trois PNG dans ``--out`` (créé si besoin) — image d'origine, heatmap,
et heatmap superposée.
"""

from __future__ import annotations

import argparse

from src.experiments.inference import load_run, predict_images, rebuild_eval_transform
from src.features.transform import load_image_rgb
from src.models.gradcam import GradCAM, default_gradcam_layer, render, resolve_layer


def _target_index(value: str | None, class_names: list[str]) -> int | None:
    """Convertit ``--target-class`` (indice entier ou nom de classe) en indice, ou ``None``."""
    if value is None:
        return None
    if value.lstrip("-").isdigit():
        return int(value)
    return list(class_names).index(value)


def main() -> None:
    """Calcule et sauvegarde le Grad-CAM d'une image pour un run donné."""
    parser = argparse.ArgumentParser(description="Grad-CAM d'un modèle entraîné sur une image.")
    parser.add_argument("--run", required=True, help="dossier du run (config.yaml + checkpoint.pt)")
    parser.add_argument("--image", required=True, help="chemin de l'image")
    parser.add_argument("--out", required=True, help="dossier de sortie (créé si besoin)")
    parser.add_argument(
        "--layer",
        default=None,
        help=(
            "nom de la couche cible (cf. model.named_modules) ; défaut : dernière Conv2d "
            "pour un CNN maison, dernier étage du backbone pour un modèle transfer"
        ),
    )
    parser.add_argument(
        "--target-class",
        default=None,
        help="classe à expliquer (nom ou indice) ; défaut : classe prédite",
    )
    args = parser.parse_args()

    config, model, task = load_run(args.run)
    transform = rebuild_eval_transform(config, args.run)
    layer = resolve_layer(model, args.layer) if args.layer else default_gradcam_layer(model)

    x = transform(load_image_rgb(args.image)).unsqueeze(0)
    target_index = _target_index(args.target_class, list(task.class_names))
    cam = GradCAM(model, layer).compute(x, target_index=target_index)
    paths = render(args.image, cam, args.out)

    pred = predict_images(model, transform, task, [args.image])[0]
    print(f"prédiction : {pred['pred_class']} ({pred['probs'][pred['pred_class']]:.3f})")
    for name, path in paths.items():
        print(f"  {name:8s} -> {path}")


if __name__ == "__main__":
    main()
