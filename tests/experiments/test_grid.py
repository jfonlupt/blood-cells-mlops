"""Tests du dépliage de grille."""

import contextlib

import pytest
from pydantic import ValidationError

from src.config import ExperimentConfig
from src.experiments import grid as grid_mod
from src.experiments.grid import _apply_overrides, expand_grid, run_grid


def test_apply_overrides_returns_validated_config():
    cfg = _apply_overrides(ExperimentConfig(), {"loss.gamma": 2.0, "train.nb_epoch": 5})
    assert cfg.loss.gamma == 2.0
    assert cfg.train.nb_epoch == 5


def test_apply_overrides_rejects_invalid_combo():
    with pytest.raises(ValidationError):
        _apply_overrides(ExperimentConfig(), {"loss.beta": 2.0})


def test_expand_grid_cartesian_product(tmp_path):
    grid = tmp_path / "g.yaml"
    grid.write_text("loss.gamma: [0.0, 2.0]\nloss.beta: [0.99, 0.999]\n")
    configs = expand_grid(ExperimentConfig(), grid)
    assert len(configs) == 4
    combos = {(c.loss.gamma, c.loss.beta) for c in configs}
    assert combos == {(0.0, 0.99), (0.0, 0.999), (2.0, 0.99), (2.0, 0.999)}


def test_expand_grid_list_of_dicts(tmp_path):
    grid = tmp_path / "g.yaml"
    grid.write_text(
        "- loss.beta: [0.9]\n"
        "- loss.gamma: [1.0, 2.0]\n"
    )
    configs = expand_grid(ExperimentConfig(), grid)
    assert len(configs) == 3


def _capture_names(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        grid_mod,
        "run_experiment",
        lambda cfg, output_root=None, name=None, evaluate_test=True: seen.append(name),
    )
    grid = tmp_path / "g.yaml"
    grid.write_text("loss.beta: [0.99, 0.999]\nloss.gamma: [1.0, 2.0]\n")
    return seen, grid


def test_run_grid_never_evaluates_test_set(tmp_path, monkeypatch):
    """Le test set ne doit jamais fuiter dans la sélection de modèle d'un grid search."""
    calls = []
    monkeypatch.setattr(
        grid_mod,
        "run_experiment",
        lambda cfg, output_root=None, name=None, evaluate_test=True: calls.append(evaluate_test),
    )
    grid = tmp_path / "g.yaml"
    grid.write_text("loss.gamma: [0.0, 2.0]\n")
    run_grid(ExperimentConfig(), grid, output_root=tmp_path)
    assert calls == [False, False]


def test_run_grid_shares_one_random_name_across_runs(tmp_path, monkeypatch):
    seen, grid = _capture_names(tmp_path, monkeypatch)
    run_grid(ExperimentConfig(), grid, output_root=tmp_path)
    assert len(seen) == 4
    assert len(set(seen)) == 1
    assert seen[0]  # non vide


def test_run_grid_propagates_explicit_name(tmp_path, monkeypatch):
    seen, grid = _capture_names(tmp_path, monkeypatch)
    run_grid(ExperimentConfig(), grid, output_root=tmp_path, name="augmentation-sweep")
    assert seen == ["augmentation-sweep"] * 4


def test_run_grid_wraps_combinations_in_one_mlflow_grid_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        grid_mod,
        "run_experiment",
        lambda cfg, output_root=None, name=None, evaluate_test=True: None,
    )
    calls = []

    @contextlib.contextmanager
    def fake_grid_run(name, task):
        calls.append((name, task))
        yield

    monkeypatch.setattr(grid_mod.tracking, "grid_run", fake_grid_run)
    grid = tmp_path / "g.yaml"
    grid.write_text("loss.gamma: [0.0, 2.0]\n")
    run_grid(ExperimentConfig(), grid, output_root=tmp_path, name="sweep")
    assert calls == [("sweep", "type")]
