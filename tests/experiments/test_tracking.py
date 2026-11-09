"""Tests de l'intégration MLflow (``log_run`` / ``grid_run``).

``tracking.mlflow`` est remplacé par un faux client (``_FakeMlflow``) pour ne
jamais dépendre d'un vrai serveur : on vérifie la forme des appels, pas le
protocole réseau.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest
import torch
import yaml
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT

from src.config import ExperimentConfig
from src.experiments import tracking
from src.experiments.artifacts import RunResult
from src.tasks import get_task


def _result(**overrides):
    defaults = {
        "config": ExperimentConfig(output={"save_checkpoint": False}),
        "task": get_task("type"),
        "histories": [{"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9], "val_f1": [0.3, 0.7]}],
        "test_metrics": {"test_loss": 0.7, "test_f1_macro": 0.5},
        "test_metrics_folds": [{"test_loss": 0.7}],
        "best_state": {},
        "name": "brave-otter",
        "duration_s": 12.3,
        "n_trainable_params": 42,
    }
    defaults.update(overrides)
    return RunResult(**defaults)


class _FakeMlflow:
    """Double de test : même forme d'API que le module ``mlflow``."""

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.experiments: list[str] = []
        self.tags: dict = {}
        self.params: dict = {}
        self.metrics: list[tuple[dict, int | None]] = []
        self.artifact_dirs: list[str] = []
        self.start_run_calls: list[dict] = []
        self._active_count = 0
        self.info = SimpleNamespace(run_id="fake-run-id")

    def _maybe_fail(self, name: str) -> None:
        if self.fail_on == name:
            raise RuntimeError(f"{name} a échoué")

    def set_experiment(self, name: str) -> None:
        self._maybe_fail("set_experiment")
        self.experiments.append(name)

    def active_run(self):
        return object() if self._active_count > 0 else None

    def start_run(self, run_name: str | None = None, nested: bool = False) -> _FakeMlflow:
        self._maybe_fail("start_run")
        self.start_run_calls.append({"run_name": run_name, "nested": nested})
        return self

    def __enter__(self) -> Self:
        self._active_count += 1
        return self

    def __exit__(self, *exc) -> bool:
        self._active_count -= 1
        return False

    def set_tags(self, tags: dict) -> None:
        self.tags.update(tags)

    def log_params(self, params: dict) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self.metrics.append((dict(metrics), step))

    def log_artifacts(self, local_dir: str) -> None:
        self.artifact_dirs.append(local_dir)


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setattr(tracking, "mlflow", fake)
    return fake


def test_enabled_is_false_without_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.enabled() is False


def test_enabled_is_true_with_tracking_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert tracking.enabled() is True


def test_log_run_is_noop_when_disabled(monkeypatch, fake_mlflow, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tracking.log_run(_result(), tmp_path / "runs" / "abcd1234", None, None)
    assert fake_mlflow.experiments == []


def test_log_run_sets_experiment_per_task(monkeypatch, fake_mlflow, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    tracking.log_run(_result(), tmp_path / "runs" / "abcd1234", None, None)
    assert fake_mlflow.experiments == ["blood_cells-type"]


def test_log_run_tags_config_hash_from_run_dir_name(monkeypatch, fake_mlflow, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    run_dir = tmp_path / "runs" / "abcd1234"
    tracking.log_run(_result(), run_dir, None, None)
    assert fake_mlflow.tags["config_hash"] == "abcd1234"


def test_log_run_logs_flattened_config_as_params(monkeypatch, fake_mlflow, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    tracking.log_run(_result(), tmp_path / "runs" / "abcd1234", None, None)
    assert fake_mlflow.params["train.nb_epoch"] == ExperimentConfig().train.nb_epoch


def test_log_run_logs_history_as_step_metrics(monkeypatch, fake_mlflow, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    tracking.log_run(_result(), tmp_path / "runs" / "abcd1234", None, None)
    steps = {step for _, step in fake_mlflow.metrics if step is not None}
    assert steps == {0, 1}
    step0 = next(m for m, step in fake_mlflow.metrics if step == 0)
    assert step0["train_loss"] == 1.0
    assert step0["val_f1"] == 0.3


def test_log_run_logs_final_aggregated_metrics(monkeypatch, fake_mlflow, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    tracking.log_run(_result(), tmp_path / "runs" / "abcd1234", None, None)
    final = next(m for m, step in fake_mlflow.metrics if step is None)
    assert final["test_f1_macro"] == 0.5
    assert final["duration_s"] == 12.3
    assert final["n_trainable_params"] == 42


def test_log_run_uploads_run_dir_as_artifacts(monkeypatch, fake_mlflow, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    run_dir = tmp_path / "runs" / "abcd1234"
    tracking.log_run(_result(), run_dir, None, None)
    assert fake_mlflow.artifact_dirs == [str(run_dir)]


def test_log_run_warns_and_does_not_raise_when_mlflow_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setattr(tracking, "mlflow", _FakeMlflow(fail_on="set_experiment"))
    with pytest.warns(UserWarning):
        tracking.log_run(_result(), tmp_path / "runs" / "abcd1234", None, None)


def test_log_run_sets_mlflow_run_id_on_result(monkeypatch, tmp_path):
    fake = _FakeMlflow()
    monkeypatch.setattr(tracking, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    result = _result()

    tracking.log_run(
        result, tmp_path / "run", torch.nn.Linear(1, 1), np.zeros((1, 1), dtype="float32")
    )

    assert result.mlflow_run_id is not None
    assert isinstance(result.mlflow_run_id, str)


def test_log_run_leaves_mlflow_run_id_none_when_tracking_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    result = _result()

    tracking.log_run(
        result, tmp_path / "run", torch.nn.Linear(1, 1), np.zeros((1, 1), dtype="float32")
    )

    assert result.mlflow_run_id is None


def test_grid_run_is_noop_when_disabled(monkeypatch, fake_mlflow):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with tracking.grid_run("sweep-name", "type"):
        pass
    assert fake_mlflow.start_run_calls == []


def test_grid_run_starts_one_parent_run(monkeypatch, fake_mlflow):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    with tracking.grid_run("sweep-name", "type"):
        pass
    assert fake_mlflow.start_run_calls == [{"run_name": "sweep-name", "nested": False}]
    assert fake_mlflow.experiments == ["blood_cells-type"]


def test_log_run_nests_under_active_grid_run(monkeypatch, fake_mlflow, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    with tracking.grid_run("sweep-name", "type"):
        tracking.log_run(_result(), tmp_path / "runs" / "abcd1234", None, None)
    nested_call = fake_mlflow.start_run_calls[-1]
    assert nested_call["nested"] is True


def test_grid_run_propagates_body_exceptions(monkeypatch, fake_mlflow):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    with pytest.raises(ValueError, match="boom from training"), tracking.grid_run("sweep-name", "type"):
        raise ValueError("boom from training")


def test_grid_run_warns_and_still_runs_body_when_mlflow_fails(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setattr(tracking, "mlflow", _FakeMlflow(fail_on="start_run"))
    executed = False
    with pytest.warns(UserWarning), tracking.grid_run("sweep-name", "type"):
        executed = True
    assert executed


def test_log_run_tags_git_commit_from_env_var(monkeypatch, fake_mlflow, tmp_path):
    """Le commit n'est jamais auto-détecté dans l'image : seule .venv (sans .git) y est copiée."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("GIT_COMMIT", "deadbeefcafe")
    tracking.log_run(_result(), tmp_path / "runs" / "abcd1234", None, None)
    assert fake_mlflow.tags[MLFLOW_GIT_COMMIT] == "deadbeefcafe"


def test_log_run_omits_git_commit_tag_without_env_var(monkeypatch, fake_mlflow, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    tracking.log_run(_result(), tmp_path / "runs" / "abcd1234", None, None)
    assert MLFLOW_GIT_COMMIT not in fake_mlflow.tags


def test_grid_run_tags_parent_run_with_git_commit(monkeypatch, fake_mlflow):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("GIT_COMMIT", "deadbeefcafe")
    with tracking.grid_run("sweep-name", "type"):
        pass
    assert fake_mlflow.tags[MLFLOW_GIT_COMMIT] == "deadbeefcafe"


class _FakeModelVersion:
    def __init__(self, run_id="run123", source="models:/foo/1"):
        self.run_id = run_id
        self.source = source


class _FakeRegistryClient:
    def __init__(self, model_version=None, raise_exc=None):
        self._mv = model_version
        self._raise_exc = raise_exc

    def get_model_version_by_alias(self, name, alias):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._mv


class _FakeModelWithFeatures(torch.nn.Module):
    """Modèle factice exposant ``.features()`` -- un vrai ``nn.Module``, nécessaire depuis
    l'ajout du garde-fou ``isinstance(model, torch.nn.Module)`` dans ``get_model_from_mlflow``
    (un objet qui n'en est pas un, comme un ``torch.export.ExportedProgram``, ferait lever ce
    garde-fou à tort ici)."""

    def features(self, x):
        return x


def test_get_model_from_mlflow_returns_model_task_transform_and_run_id(monkeypatch):
    import mlflow as real_mlflow

    fake_model = _FakeModelWithFeatures()
    monkeypatch.setattr(real_mlflow.pytorch, "load_model", lambda source: fake_model)
    monkeypatch.setattr(
        real_mlflow.artifacts,
        "load_text",
        lambda uri: yaml.safe_dump(ExperimentConfig(task="type").model_dump(mode="json")),
    )
    monkeypatch.setattr(
        real_mlflow.artifacts,
        "load_dict",
        lambda uri: {"dataset_mean": [0.5, 0.5, 0.5], "dataset_std": [0.25, 0.25, 0.25], "reinhard": None},
    )
    client = _FakeRegistryClient(model_version=_FakeModelVersion(run_id="run123"))

    transform, model, task, run_id = tracking.get_model_from_mlflow("blood_cells-type", client)

    assert run_id == "run123"
    assert model is fake_model
    assert task.name == "type"
    out = transform(torch.randint(0, 255, (3, 30, 30), dtype=torch.uint8))
    assert out.shape == (3, 224, 224)


def test_get_model_from_mlflow_raises_value_error_when_alias_missing(monkeypatch):
    import mlflow as real_mlflow
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

    exc = real_mlflow.MlflowException("introuvable", error_code=RESOURCE_DOES_NOT_EXIST)
    client = _FakeRegistryClient(raise_exc=exc)

    with pytest.raises(ValueError, match="champion"):
        tracking.get_model_from_mlflow("blood_cells-type", client, alias="champion")


def test_get_model_from_mlflow_raises_value_error_for_legacy_model_without_features(monkeypatch):
    """Garde-fou spec §6 : un champion pré-refactor (multi_archi/baseline/dataset) charge
    sans erreur (pickle) mais n'a pas ``.features`` -- doit être détecté ici plutôt que
    planter plus tard dans un ``forward()`` avec un ``AttributeError``/``TypeError`` opaque.
    """
    import mlflow as real_mlflow

    # Un vrai nn.Module (donc passe le garde isinstance) mais sans .features, comme un
    # vieux pickle pré-refactor.
    legacy_model = torch.nn.Module()
    monkeypatch.setattr(real_mlflow.pytorch, "load_model", lambda source: legacy_model)
    client = _FakeRegistryClient(model_version=_FakeModelVersion(run_id="run123"))

    with pytest.raises(ValueError, match="re-logger"):
        tracking.get_model_from_mlflow("blood_cells-type", client)


def test_get_model_from_mlflow_raises_value_error_when_not_nn_module(monkeypatch):
    """mlflow>=3 sérialise en "pt2" (torch.export) hors Databricks par défaut : load_model
    renvoie alors un ``torch.export.ExportedProgram``, pas un ``nn.Module`` -- doit être
    détecté ici avec un message clair plutôt que planter loin d'ici dans un ``.eval()``/
    ``.features()`` avec une trace torch illisible (voir log_run : serialization_format="pickle").
    """
    import mlflow as real_mlflow

    exported_program = object()  # objet quelconque non-nn.Module, comme un ExportedProgram
    monkeypatch.setattr(real_mlflow.pytorch, "load_model", lambda source: exported_program)
    client = _FakeRegistryClient(model_version=_FakeModelVersion(run_id="run123"))

    with pytest.raises(ValueError, match="torch.nn.Module"):
        tracking.get_model_from_mlflow("blood_cells-type", client)
