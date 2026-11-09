"""Tests de la fabrique de modèles."""

import pytest
import torch
from torch import nn

from src.config import ArchitectureConfig, BlockConfig, ModelConfig, TransferConfig
from src.models import build_model
from src.tasks import get_task


def test_baseline_output_dim_matches_task():
    model = build_model(ModelConfig(name="baseline"), get_task("type"))
    assert isinstance(model, nn.Module)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)


def test_dataset_task_builds_single_logit_model():
    model = build_model(ModelConfig(name="dataset"), get_task("dataset"))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_dataset_model_respects_img_size():
    model = build_model(ModelConfig(name="dataset"), get_task("dataset"), img_size=(224, 224))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_transfer_builds_from_config():
    from src.config import TransferConfig

    cfg = ModelConfig(name="transfer", transfer=TransferConfig(backbone="resnet18", pretrained=False))
    model = build_model(cfg, get_task("type"))
    assert isinstance(model, nn.Module)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)


def test_models_are_plain_nn_module():
    from src.models.baseline import BaselineClassifier

    assert BaselineClassifier.__mro__[1] is nn.Module


@pytest.mark.parametrize(
    ("name", "task_name"),
    [("baseline", "type"), ("dataset", "dataset"), ("multi_archi", "type"), ("transfer", "type")],
)
def test_every_model_exposes_features_before_head(name, task_name):
    """Filet de régression : les 4 modèles exposent .features(), testé en détail par modèle
    dans test_baseline.py/test_dataset_model.py/test_multi_archi.py/test_transfer.py
    (égalité classifier(features(x)) == forward(x)) — ici on verrouille juste le contrat
    commun sur toute la fabrique build_model.
    """
    kwargs = {}
    if name == "multi_archi":
        kwargs["architecture"] = ArchitectureConfig(
            blocks=[BlockConfig(depth=1, channels=8, kernel=3)], classifier=[16]
        )
    if name == "transfer":
        kwargs["transfer"] = TransferConfig(backbone="resnet18", pretrained=False)

    model = build_model(ModelConfig(name=name, **kwargs), get_task(task_name))
    assert hasattr(model, "features")
    x = torch.randn(2, 3, 224, 224)
    feats = model.features(x)
    assert feats.shape[0] == 2
    assert feats.ndim == 2
    assert feats.shape[1] > 0
