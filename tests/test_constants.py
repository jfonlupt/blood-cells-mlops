"""Tests de résolution du répertoire de données (local vs conteneur Docker)."""

from __future__ import annotations

import importlib
from pathlib import Path

from src import constants as constants_mod


def _reload_constants():
    return importlib.reload(constants_mod)


def test_data_path_defaults_to_base_dir_data_without_env_var(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    mod = _reload_constants()
    assert mod.DATA_PATH == mod.BASE_DIR / "data"


def test_data_path_respects_data_dir_env_var(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/app/data")
    try:
        mod = _reload_constants()
        assert Path("/app/data") == mod.DATA_PATH
    finally:
        monkeypatch.delenv("DATA_DIR", raising=False)
        _reload_constants()


def test_dataset_paths_follow_data_dir_env_var(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/app/data")
    try:
        mod = _reload_constants()
        assert mod.DATASET_PATHS["barcelona"] == Path("/app/data/PBC_dataset_normal_DIB")
    finally:
        monkeypatch.delenv("DATA_DIR", raising=False)
        _reload_constants()


def test_new_drift_dataset_keys_resolve_under_data_dir(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/app/data")
    try:
        mod = _reload_constants()
        assert mod.DATASET_PATHS["munich_train"] == Path("/app/data/train/munich")
        assert mod.DATASET_PATHS["munich_eval"] == Path("/app/data/test/munich")
        assert mod.DATASET_PATHS["data_drift_train"] == Path("/app/data/train/data_drift")
        assert mod.DATASET_PATHS["data_drift_eval"] == Path("/app/data/test/data_drift")
        assert "data_drift" not in mod.DATASET_PATHS
        assert mod.DATASET_PATHS["munich"] == Path("/app/data/PKG-AML-Cytomorphology_LMU")
    finally:
        monkeypatch.delenv("DATA_DIR", raising=False)
        _reload_constants()
