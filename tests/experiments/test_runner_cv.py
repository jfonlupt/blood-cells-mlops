"""Assemblage du chemin validation croisée du runner, sans entraînement réel."""

import json

import pandas as pd
import pytest
import torch

from src.config import ExperimentConfig
from src.experiments import runner
from src.experiments.runner import _best_fold_index, run_experiment


class _FakeTrainer:
    def __init__(self, model, *args, **kwargs):
        self.model = model

    def fit(self, train_ds, val_ds=None):
        return {"train_loss": [1.0], "val_loss": [0.9], "lr": [0.01]}

    def evaluate(self, dataset, metrics=None, prefix="test"):
        return {f"{prefix}_loss": 0.5, f"{prefix}_f1_macro": 0.6, f"{prefix}_mcc": 0.4}


def test_best_fold_index_picks_lowest_val_loss():
    histories = [
        {"val_loss": [0.9, 0.7, 0.75]},  # meilleure epoch : 0.7
        {"val_loss": [0.5, 0.4]},  # meilleure epoch : 0.4 -> ce fold gagne
        {"val_loss": [0.6]},
    ]
    assert _best_fold_index(histories) == 1


def test_best_fold_index_single_fold_is_trivial():
    assert _best_fold_index([{"val_loss": [1.2, 0.3]}]) == 0


def test_best_fold_index_falls_back_to_zero_without_val_loss():
    """``train.validation=\"none\"`` : pas de val_loss, aucun critère de comparaison."""
    assert _best_fold_index([{"train_loss": [1.0]}]) == 0


class _FakeTransformBuilder:
    def __init__(self, *args, **kwargs):
        self.dataset_mean = torch.tensor([0.5, 0.5, 0.5])
        self.dataset_std = torch.tensor([0.25, 0.25, 0.25])

    def build(self, fit_df, *, training):
        return lambda x: x


def _fake_df(n=40):
    rows = {
        "path": [f"/nonexistent/img_{i}.png" for i in range(n)],
        "label": [i % 10 for i in range(n)],
        "type": ["Neutrophil"] * n,
        "type_dataset": ["Neutrophil_barcelona"] * n,
        "dataset": ["barcelona"] * n,
        "binary_criticality": ["normal"] * n,
        "criticality": ["normal"] * n,
    }
    return pd.DataFrame(rows)


def test_run_experiment_cross_validation_assembly(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "Trainer", _FakeTrainer)
    monkeypatch.setattr(runner, "TransformBuilder", _FakeTransformBuilder)
    monkeypatch.setattr(runner, "create_dataframe", lambda data_cfg, task: _fake_df())
    monkeypatch.setattr(runner, "build_drift_reference", lambda *a, **k: None)

    cfg = ExperimentConfig.model_validate(
        {
            "task": "type",
            "seed": 0,
            "data": {"stratify": "no", "test_size": 0.25},
            "transform": {"reinhard": False, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 1, "batch_size": 8, "validation": "cross", "cross_val_folds": 2},
            "output": {"root": str(tmp_path), "save_figures": False, "save_checkpoint": False},
        }
    )

    result = run_experiment(cfg, output_root=tmp_path)

    assert result.status == "ok"
    assert "test_f1_macro_mean" in result.test_metrics
    assert "test_f1_macro_std" in result.test_metrics
    assert len(result.histories) == 2
    assert result.run_dir.exists()
    assert (result.run_dir / "config.yaml").exists()
    assert (result.run_dir / "metrics_test.json").exists()

    history = json.loads((result.run_dir / "history.json").read_text())
    assert isinstance(history, list)
    assert len(history) == 2


def test_run_experiment_cv_keeps_best_fold_not_first(tmp_path, monkeypatch):
    """``best_state`` doit venir du fold au meilleur val_loss, pas du premier entraîné."""
    val_losses_per_fold = [0.9, 0.3, 0.6]  # le fold 1 (index 1) est le meilleur
    call_count = {"n": 0}

    class _FoldAwareTrainer(_FakeTrainer):
        def fit(self, train_ds, val_ds=None):
            i = call_count["n"]
            call_count["n"] += 1
            # marque le modèle pour pouvoir l'identifier après coup dans best_state
            with torch.no_grad():
                for p in self.model.parameters():
                    p.fill_(float(i))
                    break
            return {"train_loss": [1.0], "val_loss": [val_losses_per_fold[i]], "lr": [0.01]}

    monkeypatch.setattr(runner, "Trainer", _FoldAwareTrainer)
    monkeypatch.setattr(runner, "TransformBuilder", _FakeTransformBuilder)
    monkeypatch.setattr(runner, "create_dataframe", lambda data_cfg, task: _fake_df())
    monkeypatch.setattr(runner, "build_drift_reference", lambda *a, **k: None)

    cfg = ExperimentConfig.model_validate(
        {
            "task": "type",
            "seed": 0,
            "data": {"stratify": "no", "test_size": 0.25},
            "transform": {"reinhard": False, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 1, "batch_size": 8, "validation": "cross", "cross_val_folds": 3},
            "output": {"root": str(tmp_path), "save_figures": False, "save_checkpoint": False},
        }
    )
    result = run_experiment(cfg, output_root=tmp_path)

    first_param = next(iter(result.best_state.values()))
    assert torch.allclose(first_param, torch.full_like(first_param, 1.0))


def _fake_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "Trainer", _FakeTrainer)
    monkeypatch.setattr(runner, "TransformBuilder", _FakeTransformBuilder)
    monkeypatch.setattr(runner, "create_dataframe", lambda data_cfg, task: _fake_df())
    monkeypatch.setattr(runner, "build_drift_reference", lambda *a, **k: None)
    return ExperimentConfig.model_validate(
        {
            "task": "type",
            "data": {"stratify": "no", "test_size": 0.25},
            "transform": {"reinhard": False, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 1, "batch_size": 8, "validation": "holdout", "val_size": 0.25},
            "output": {"root": str(tmp_path), "save_figures": False, "save_checkpoint": False},
        }
    )


def test_run_experiment_assigns_random_name_when_none(tmp_path, monkeypatch):
    cfg = _fake_holdout(tmp_path, monkeypatch)
    result = run_experiment(cfg, output_root=tmp_path)
    assert result.name  # nom aléatoire non vide
    meta = json.loads((result.run_dir / "meta.json").read_text())
    assert meta["name"] == result.name


def test_run_experiment_uses_provided_name(tmp_path, monkeypatch):
    cfg = _fake_holdout(tmp_path, monkeypatch)
    result = run_experiment(cfg, output_root=tmp_path, name="my-experiment")
    assert result.name == "my-experiment"
    df = pd.read_csv(tmp_path / "leaderboard.csv")
    assert df["name"].iloc[0] == "my-experiment"


class _InterruptingTrainer(_FakeTrainer):
    """Simule un Ctrl+C pendant ``fit`` : ``history`` porte l'historique partiel."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.history = {"train_loss": [1.0], "val_loss": [0.9], "lr": [0.01]}

    def fit(self, train_ds, val_ds=None):
        raise KeyboardInterrupt


def test_run_experiment_saves_partial_result_on_keyboard_interrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "Trainer", _InterruptingTrainer)
    monkeypatch.setattr(runner, "TransformBuilder", _FakeTransformBuilder)
    monkeypatch.setattr(runner, "create_dataframe", lambda data_cfg, task: _fake_df())
    monkeypatch.setattr(runner, "build_drift_reference", lambda *a, **k: None)

    cfg = ExperimentConfig.model_validate(
        {
            "task": "type",
            "data": {"stratify": "no", "test_size": 0.25},
            "transform": {"reinhard": False, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 5, "batch_size": 8, "validation": "holdout", "val_size": 0.25},
            "output": {"root": str(tmp_path), "save_figures": False, "save_checkpoint": False},
        }
    )

    with pytest.raises(KeyboardInterrupt):
        run_experiment(cfg, output_root=tmp_path)

    df = pd.read_csv(tmp_path / "leaderboard.csv")
    assert df["status"].iloc[0] == "interrupted"
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "history.json").exists()


def test_run_experiment_saves_preprocessing_stats(tmp_path, monkeypatch):
    cfg = _fake_holdout(tmp_path, monkeypatch)  # transform.reinhard = False
    result = run_experiment(cfg, output_root=tmp_path)
    pp = json.loads((result.run_dir / "preprocessing.json").read_text())
    assert pp["dataset_mean"] == [0.5, 0.5, 0.5]
    assert pp["dataset_std"] == [0.25, 0.25, 0.25]
    assert pp["reinhard"] is None  # désactivé dans la config


def test_run_experiment_saves_reinhard_stats_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "Trainer", _FakeTrainer)
    monkeypatch.setattr(runner, "TransformBuilder", _FakeTransformBuilder)
    monkeypatch.setattr(runner, "create_dataframe", lambda data_cfg, task: _fake_df())
    monkeypatch.setattr(runner, "build_drift_reference", lambda *a, **k: None)

    class _FakeNormalizer:
        target_means = torch.tensor([70.0, 5.0, 10.0])
        target_stds = torch.tensor([15.0, 4.0, 6.0])

    monkeypatch.setattr(runner, "build_reinhard", lambda *a, **k: _FakeNormalizer())
    cfg = ExperimentConfig.model_validate(
        {
            "task": "type",
            "data": {"stratify": "no", "test_size": 0.25},
            "transform": {"reinhard": True, "clahe": False, "jpeg": False},
            "train": {"nb_epoch": 1, "batch_size": 8, "validation": "holdout", "val_size": 0.25},
            "output": {"root": str(tmp_path), "save_figures": False, "save_checkpoint": False},
        }
    )
    result = run_experiment(cfg, output_root=tmp_path)
    pp = json.loads((result.run_dir / "preprocessing.json").read_text())
    assert pp["reinhard"]["target_means"] == [70.0, 5.0, 10.0]
    assert pp["reinhard"]["target_stds"] == [15.0, 4.0, 6.0]
