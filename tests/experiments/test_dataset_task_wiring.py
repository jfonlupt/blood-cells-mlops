"""Câblage de la tâche ``dataset`` (BCE / 1 logit / label float) — sans entraînement."""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from src.config import LossConfig, ModelConfig
from src.features.dataset import BloodCellDataset
from src.models import build_model
from src.models.losses import build_loss
from src.models.metrics import build_metrics
from src.tasks import get_task


def _df(n=8):
    return pd.DataFrame({"label": [0, 1] * (n // 2), "dataset": "barcelona"})


def test_dataset_model_is_single_logit_module():
    model = build_model(ModelConfig(name="dataset"), get_task("dataset"))
    assert isinstance(model, nn.Module)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_dataset_loss_is_bce_with_logits():
    loss = build_loss(LossConfig(), get_task("dataset"), _df())
    assert isinstance(loss, nn.BCEWithLogitsLoss)


def test_dataset_metrics_have_acc():
    mc = build_metrics(get_task("dataset"))
    assert "acc" in mc


def test_dataset_label_is_float32_vector1(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "img.png"
    Image.fromarray(np.uint8(rng.random((20, 20, 3)) * 255)).save(p)
    df = pd.DataFrame({"path": [str(p)], "label": [1], "dataset": ["barcelona"]})
    _, label = BloodCellDataset(df, get_task("dataset"))[0]
    assert label.dtype == torch.float32
    assert label.shape == (1,)
