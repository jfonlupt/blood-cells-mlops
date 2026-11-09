"""Fabrique de modèles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.baseline import BaselineClassifier
from src.models.dataset_model import DatasetClassifier
from src.models.multi_archi_model import MultiArchiModel

if TYPE_CHECKING:
    from torch import nn

    from src.config import ModelConfig
    from src.tasks import TaskSpec


def build_model(
    model_cfg: ModelConfig, task: TaskSpec, img_size: tuple[int, int] = (224, 224),
) -> nn.Module:
    """Instancie le modèle correspondant à la config et à la tâche.

    Args:
        model_cfg: Configuration du modèle (``name``).
        task: Spécification de la tâche.
        img_size: Taille ``(H, W)`` des images en entrée. Utilisée par
            ``DatasetClassifier`` (taille du ``Flatten``) ; ``BaselineClassifier``
            est câblé en dur pour ``(224, 224)``.

    Returns:
        Le ``nn.Module`` instancié.
    """
    if model_cfg.name == "dataset":
        return DatasetClassifier(img_size=tuple(img_size))
    if model_cfg.name == "baseline":
        return BaselineClassifier(task.num_classes)
    if model_cfg.name == "multi_archi":
        return MultiArchiModel(
            task.num_classes,
            img_size=tuple(img_size),
            blocks=model_cfg.architecture.blocks,
            classifier=model_cfg.architecture.classifier,
            use_batchnorm=model_cfg.architecture.use_batchnorm,
            use_gap=model_cfg.architecture.use_gap
                               )
    if model_cfg.name == "transfer":
        from src.models.transfer import TransferModel

        return TransferModel(model_cfg.transfer, task.num_classes)

    raise ValueError(f"modèle inconnu : {model_cfg.name!r}")
