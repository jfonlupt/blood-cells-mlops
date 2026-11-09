"""Tests des artefacts de run (dossier + leaderboard)."""

import json

import pandas as pd
import pytest
import torch

from src.config import ExperimentConfig, load_config
from src.experiments.artifacts import (
    RunResult,
    append_leaderboard,
    dirname_for,
    flatten,
    write_run_dir,
)
from src.experiments.drift_reference import DriftReference
from src.tasks import get_task


def test_flatten_nested_dict():
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}


def test_dirname_is_deterministic_for_same_config():
    assert dirname_for(ExperimentConfig()) == dirname_for(ExperimentConfig())


def _result(tmp_path, validation="holdout"):
    cfg = ExperimentConfig(train={"validation": validation, "nb_epoch": 2})
    hist = {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9], "lr": [0.01, 0.01]}
    test_metrics = {"test_loss": 0.7, "test_f1_macro": 0.5, "test_mcc": 0.4}
    return RunResult(
        config=cfg,
        task=get_task("type"),
        histories=[hist],
        test_metrics=test_metrics,
        test_metrics_folds=[test_metrics],
        best_state={"w": torch.zeros(1)},
        run_dir=None,
        duration_s=1.0,
    )


def test_run_dir_is_nested_under_runs_subdir(tmp_path):
    run_dir = write_run_dir(_result(tmp_path), tmp_path)
    assert run_dir.parent == tmp_path / "runs"


def test_write_run_dir_writes_meta_json_with_name(tmp_path):
    result = _result(tmp_path)
    result.name = "brave-otter"
    run_dir = write_run_dir(result, tmp_path)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["name"] == "brave-otter"


def test_append_leaderboard_name_column_follows_run_dir(tmp_path):
    lb = tmp_path / "leaderboard.csv"
    result = _result(tmp_path)
    result.name = "calm-lynx"
    result.run_dir = tmp_path / "runs" / "abcd1234"
    append_leaderboard(lb, result)
    df = pd.read_csv(lb)
    assert list(df.columns)[:2] == ["run_dir", "name"]
    assert df["name"].iloc[0] == "calm-lynx"


def test_write_run_dir_writes_preprocessing_json_when_set(tmp_path):
    result = _result(tmp_path)
    result.preprocessing = {
        "dataset_mean": [0.71, 0.62, 0.66],
        "dataset_std": [0.18, 0.19, 0.13],
        "reinhard": {"target_means": [70.0, 5.0, 10.0], "target_stds": [15.0, 4.0, 6.0]},
    }
    run_dir = write_run_dir(result, tmp_path)
    pp = json.loads((run_dir / "preprocessing.json").read_text())
    assert pp["dataset_mean"] == [0.71, 0.62, 0.66]
    assert pp["reinhard"]["target_stds"] == [15.0, 4.0, 6.0]


def test_write_run_dir_skips_preprocessing_json_when_absent(tmp_path):
    run_dir = write_run_dir(_result(tmp_path), tmp_path)
    assert not (run_dir / "preprocessing.json").exists()


def test_write_run_dir_creates_expected_files(tmp_path):
    result = _result(tmp_path)
    run_dir = write_run_dir(result, tmp_path)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "metrics_test.json").exists()
    reloaded = load_config(run_dir / "config.yaml")
    assert reloaded.train.nb_epoch == 2


def test_append_leaderboard_adds_rows_without_duplicating_header(tmp_path):
    lb = tmp_path / "leaderboard.csv"
    append_leaderboard(lb, _result(tmp_path))
    append_leaderboard(lb, _result(tmp_path))
    df = pd.read_csv(lb)
    assert len(df) == 2
    assert "loss.name" in df.columns
    assert "test_f1_macro" in df.columns


def test_append_leaderboard_emits_spec_columns(tmp_path):
    lb = tmp_path / "leaderboard.csv"
    result = _result(tmp_path)
    result.histories = [
        {
            "train_loss": [1.0, 0.8],
            "val_loss": [1.1, 0.9],
            "val_f1": [0.3, 0.7],
            "lr": [0.01, 0.01],
        }
    ]
    result.test_metrics_raw = {
        "test_loss": 0.7,
        "test_f1_macro": 0.5,
        "test_recall_per_class": [0.9] * 10,
    }
    append_leaderboard(lb, result)
    df = pd.read_csv(lb)
    assert "min_critical_recall" in df.columns
    assert "val_f1_macro" in df.columns
    # val_f1 à l'epoch de val_loss minimale (index 1)
    assert df["val_f1_macro"].iloc[0] == pytest.approx(0.7)
    assert df["min_critical_recall"].iloc[0] == pytest.approx(0.9)


def test_append_leaderboard_aligns_differing_columns(tmp_path):
    lb = tmp_path / "leaderboard.csv"
    holdout = _result(tmp_path)
    holdout.test_metrics = {"test_f1_macro": 0.42}
    append_leaderboard(lb, holdout)
    cv = _result(tmp_path)
    cv.test_metrics = {"test_f1_macro_mean": 0.55}
    append_leaderboard(lb, cv)
    df = pd.read_csv(lb)
    assert len(df) == 2
    assert {"test_f1_macro", "test_f1_macro_mean"} <= set(df.columns)
    assert df["test_f1_macro"].iloc[0] == pytest.approx(0.42)
    assert df["test_f1_macro_mean"].iloc[1] == pytest.approx(0.55)
    assert pd.isna(df["test_f1_macro"].iloc[1])


def test_write_run_dir_writes_drift_reference_descriptors(tmp_path):
    result = _result(tmp_path)
    result.drift_reference = DriftReference(
        descriptors=pd.DataFrame({"path": ["a.png"], "brightness": [0.5]}), embeddings=None
    )
    run_dir = write_run_dir(result, tmp_path)
    df = pd.read_parquet(run_dir / "drift_reference" / "descriptors.parquet")
    assert df["brightness"].iloc[0] == 0.5
    assert not (run_dir / "drift_reference" / "embeddings.parquet").exists()


def test_write_run_dir_writes_drift_reference_embeddings_when_present(tmp_path):
    result = _result(tmp_path)
    result.drift_reference = DriftReference(
        descriptors=pd.DataFrame({"path": ["a.png"], "brightness": [0.5]}),
        embeddings=pd.DataFrame({"path": ["a.png"], "emb_0": [1.0]}),
    )
    run_dir = write_run_dir(result, tmp_path)
    df = pd.read_parquet(run_dir / "drift_reference" / "embeddings.parquet")
    assert df["emb_0"].iloc[0] == 1.0


def test_write_run_dir_skips_drift_reference_dir_when_absent(tmp_path):
    run_dir = write_run_dir(_result(tmp_path), tmp_path)
    assert not (run_dir / "drift_reference").exists()
