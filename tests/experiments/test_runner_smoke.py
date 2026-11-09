"""Test d'intégration léger du runner (nécessite les données sur disque)."""

import pandas as pd
import pytest

from src.config import ExperimentConfig
from src.constants import DATASET_PATHS
from src.experiments import runner
from src.experiments.runner import run_experiment

pytestmark = pytest.mark.skipif(
    not DATASET_PATHS["barcelona"].exists(), reason="données absentes"
)


def test_holdout_run_produces_artifacts(tmp_path):
    cfg = ExperimentConfig.model_validate(
        {
            "task": "type",
            "seed": 0,
            "data": {"n_images": 120, "stratify": "type", "test_size": 0.25},
            "transform": {"reinhard": False, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 1, "batch_size": 16, "validation": "holdout", "val_size": 0.25},
            "output": {"root": str(tmp_path), "save_checkpoint": True},
        }
    )
    result = run_experiment(cfg, output_root=tmp_path)
    assert result.status == "ok"
    assert result.run_dir.exists()
    assert (result.run_dir / "config.yaml").exists()
    assert (result.run_dir / "metrics_test.json").exists()
    assert "test_loss" in result.test_metrics
    assert (tmp_path / "leaderboard.csv").exists()
    assert len(pd.read_csv(tmp_path / "leaderboard.csv")) == 1


def test_evaluate_test_false_skips_test_evaluation(tmp_path):
    """Utilisé par le grid search : pas d'évaluation test (évite la fuite dans la sélection)."""
    cfg = ExperimentConfig.model_validate(
        {
            "task": "type",
            "seed": 0,
            "data": {"n_images": 120, "stratify": "type", "test_size": 0.25},
            "transform": {"reinhard": False, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 1, "batch_size": 16, "validation": "holdout", "val_size": 0.25},
            "output": {"root": str(tmp_path), "save_checkpoint": False, "save_figures": False},
        }
    )
    result = run_experiment(cfg, output_root=tmp_path, evaluate_test=False)
    assert result.status == "ok"
    assert result.test_metrics == {}
    assert result.test_metrics_raw is None
    leaderboard = pd.read_csv(tmp_path / "leaderboard.csv")
    assert "test_loss" not in leaderboard.columns or leaderboard["test_loss"].isna().all()


def test_holdout_run_writes_drift_reference(tmp_path):
    cfg = ExperimentConfig.model_validate(
        {
            "task": "type",
            "seed": 0,
            "data": {"n_images": 120, "stratify": "type", "test_size": 0.25},
            "transform": {"reinhard": False, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 1, "batch_size": 16, "validation": "holdout", "val_size": 0.25},
            "output": {"root": str(tmp_path), "save_checkpoint": True},
        }
    )
    result = run_experiment(cfg, output_root=tmp_path)
    assert (result.run_dir / "drift_reference" / "descriptors.parquet").exists()
    assert (result.run_dir / "drift_reference" / "embeddings.parquet").exists()


def test_run_experiment_calls_mlflow_tracking_hook(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner.tracking, "log_run", lambda result, run_dir, model, x: calls.append((result, run_dir))
    )
    cfg = ExperimentConfig.model_validate(
        {
            "task": "type",
            "seed": 0,
            "data": {"n_images": 120, "stratify": "type", "test_size": 0.25},
            "transform": {"reinhard": False, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 1, "batch_size": 16, "validation": "holdout", "val_size": 0.25},
            "output": {"root": str(tmp_path), "save_checkpoint": False, "save_figures": False},
        }
    )
    result = run_experiment(cfg, output_root=tmp_path)
    assert len(calls) == 1
    logged_result, logged_run_dir = calls[0]
    assert logged_result is result
    assert logged_run_dir == result.run_dir
