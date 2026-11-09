"""Tests du Dataset image piloté par TaskSpec."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.features.dataset import BloodCellDataset, DatasetTransform
from src.tasks import get_task


@pytest.fixture
def tiny_df(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(np.uint8(rng.random((20, 20, 3)) * 255)).save(p)
        rows.append({"path": str(p), "label": i % 2})
    return pd.DataFrame(rows)


def test_label_dtype_scalar_for_multiclass(tiny_df):
    ds = BloodCellDataset(tiny_df, get_task("type"))
    _, label = ds[0]
    assert label.dtype == torch.long
    assert label.ndim == 0


def test_label_shape_vector1_for_dataset_task(tiny_df):
    ds = BloodCellDataset(tiny_df, get_task("dataset"))
    _, label = ds[0]
    assert label.dtype == torch.float32
    assert label.shape == (1,)


def test_transform_is_applied(tiny_df):
    ds = BloodCellDataset(tiny_df, get_task("type"), transform=lambda x: torch.zeros(3, 8, 8))
    image, _ = ds[0]
    assert image.shape == (3, 8, 8)


def test_dataset_transform_wraps_subset(tiny_df):
    base = BloodCellDataset(tiny_df, get_task("type"))
    wrapped = DatasetTransform(base, transform=lambda x: x * 0)
    image, _ = wrapped[0]
    assert torch.equal(image, torch.zeros_like(image))
