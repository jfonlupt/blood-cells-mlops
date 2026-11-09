"""Tests de construction du DataFrame d'images."""

import pytest

from src.config import DataConfig
from src.constants import DATASET_PATHS
from src.tasks import get_task

pytestmark = pytest.mark.skipif(
    not DATASET_PATHS["barcelona"].exists(), reason="données absentes"
)


def test_create_dataframe_columns_and_labels():
    from src.features.dataframe import create_dataframe

    df = create_dataframe(DataConfig(n_images=200, stratify="type"), get_task("type"))
    assert not df.empty
    assert {"path", "dataset", "label", "type", "criticality"} <= set(df.columns)
    assert df["label"].between(0, 9).all()
    assert df["label"].notna().all()


def test_create_dataframe_task_criticality_labels_in_range():
    from src.features.dataframe import create_dataframe

    df = create_dataframe(DataConfig(n_images=200, stratify="no"), get_task("criticality"))
    assert df["label"].between(0, 2).all()


def test_n_images_zero_returns_full_dataset():
    from src.features.dataframe import create_dataframe

    df = create_dataframe(DataConfig(n_images=0), get_task("type"))
    assert len(df) > 1000
