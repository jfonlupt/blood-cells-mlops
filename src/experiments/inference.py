"""Inférence avec un modèle déjà entraîné : recharge un run, refait le prétraitement, prédit.

Le transform d'évaluation est reconstruit sans relire les données quand le run
contient un ``preprocessing.json`` (moyenne/écart-type + stats Reinhard) ; sinon
on rejoue la séquence de ``runner.py`` à partir du ``seed`` (mean/std ré-estimés,
pas au bit près — acceptable pour un diagnostic).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchstain

from src.config import load_config
from src.experiments.runner import seed_everything
from src.features.dataframe import create_dataframe
from src.features.stats import build_reinhard
from src.features.transform import build_transform, load_image_rgb
from src.models import build_model
from src.tasks import get_task
from src.training.components import TransformBuilder, split_off_test

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from src.config import ExperimentConfig
    from src.tasks import TaskSpec


def load_run(run_dir: str | Path, device: str | torch.device = "cpu"):
    """Recharge un run : config, modèle (poids chargés, ``.eval()``), tâche.

    Args:
        run_dir: Dossier du run (``config.yaml`` + ``checkpoint.pt``).
        device: Device cible du modèle.

    Returns:
        ``(config, model, task)``.
    """
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    task = get_task(config.task)
    model = build_model(config.model, task, img_size=config.transform.img_size)
    state = torch.load(run_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return config, model, task


def _reinhard_from_stats(stats: dict):
    """Recrée un ``ReinhardNormalizer`` à partir de ses seules stats cible Lab."""
    normalizer = torchstain.normalizers.ReinhardNormalizer(backend="torch", method=None)
    normalizer.target_means = torch.tensor(stats["target_means"], dtype=torch.float32)
    normalizer.target_stds = torch.tensor(stats["target_stds"], dtype=torch.float32)
    return normalizer


def rebuild_eval_transform(config: ExperimentConfig, run_dir: str | Path):
    """Reconstruit le ``v2.Compose`` d'évaluation du run.

    Chemin rapide si ``run_dir/preprocessing.json`` existe (aucune donnée
    requise) ; sinon rejoue ``seed → create_dataframe → split_off_test →
    build_reinhard → TransformBuilder.build(train_df, training=False)``.
    """
    run_dir = Path(run_dir)
    pp_path = run_dir / "preprocessing.json"

    if pp_path.exists():
        pp = json.loads(pp_path.read_text())
        normalizer = _reinhard_from_stats(pp["reinhard"]) if pp.get("reinhard") else None
        return build_transform(
            config.transform,
            training=False,
            reinhard_normalizer=normalizer,
            dataset_mean=torch.tensor(pp["dataset_mean"]),
            dataset_std=torch.tensor(pp["dataset_std"]),
        )

    seed_everything(config.seed)
    task = get_task(config.task)
    df = create_dataframe(config.data, task)
    train_df, _ = split_off_test(df, config.data, seed=config.seed)
    normalizer = (
        build_reinhard(train_df, config.transform, seed=config.seed)
        if config.transform.reinhard
        else None
    )
    return TransformBuilder(config.transform, task, reinhard_normalizer=normalizer).build(
        train_df, training=False
    )


def predict_images(
    model: Callable,
    transform: Callable,
    task: TaskSpec,
    image_paths: Sequence[str | Path],
    image_files: Sequence[bytes] | None = None,
    device: str | torch.device = "cpu",
) -> list[dict]:
    """Prédit la classe de chaque image.

    Returns:
        Une liste de ``{"path", "pred_index", "pred_class", "probs": {classe: p}}``.
        ``probs`` vient d'un softmax (multiclasse) ou d'un sigmoïde (tâche binaire).
    """
    results: list[dict] = []
    image_files = image_files or image_paths
    for path, file in zip(image_paths, image_files):
        x = transform(load_image_rgb(file)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)

        if task.kind == "binary":
            p1 = torch.sigmoid(logits).reshape(-1)[0].item()
            probs = {task.class_names[0]: 1.0 - p1, task.class_names[1]: p1}
            index = int(p1 > 0.5)
        else:
            vec = torch.softmax(logits.reshape(-1), dim=0)
            probs = {name: vec[i].item() for i, name in enumerate(task.class_names)}
            index = int(vec.argmax())

        results.append(
            {
                "path": str(path),
                "pred_index": index,
                "pred_class": task.class_names[index],
                "probs": probs,
            }
        )
    return results


def run_inference(
    run_dir: str | Path,
    image_paths: Sequence[str | Path],
    image_files: Sequence[bytes] | None = None,
    output: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> list[dict]:
    """Recharge le run, reconstruit le transform, prédit ; écrit ``predictions.json`` si ``output``."""
    config, model, task = load_run(run_dir, device=device)
    transform = rebuild_eval_transform(config, run_dir)
    predictions = predict_images(model, transform, task, image_paths, image_files, device=device)

    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "predictions.json").write_text(json.dumps(predictions, indent=2))

    return predictions
