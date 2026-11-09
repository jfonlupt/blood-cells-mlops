"""Tests des statistiques de normalisation."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.config import TransformConfig
from src.tasks import get_task


@pytest.fixture
def tiny_df(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(8):
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(np.uint8(rng.random((30, 30, 3)) * 255)).save(p)
        rows.append({"path": str(p), "label": i % 2, "dataset": "barcelona"})
    return pd.DataFrame(rows)


def test_compute_dataset_mean_std_finite(tiny_df):
    from src.features.stats import compute_dataset_mean_std

    cfg = TransformConfig(reinhard=False, clahe=False, jpeg=False)
    mean, std = compute_dataset_mean_std(tiny_df, cfg, get_task("type"), n_batches=2, batch_size=4)
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert (std >= 0).all()


def test_compute_dataset_mean_std_ignores_norm_stats(tiny_df):
    """La sonde mesure les stats brutes : norm_stats='imagenet' ne doit pas la biaiser."""
    from src.features.stats import compute_dataset_mean_std

    base = {"reinhard": False, "clahe": False, "jpeg": False}
    task = get_task("type")
    m_dataset, s_dataset = compute_dataset_mean_std(
        tiny_df, TransformConfig(**base, norm_stats="dataset"), task, n_batches=2, batch_size=4
    )
    m_imagenet, s_imagenet = compute_dataset_mean_std(
        tiny_df, TransformConfig(**base, norm_stats="imagenet"), task, n_batches=2, batch_size=4
    )
    assert torch.allclose(m_dataset, m_imagenet)
    assert torch.allclose(s_dataset, s_imagenet)
