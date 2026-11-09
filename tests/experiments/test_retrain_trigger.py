"""Déclenchement de réentraînement : requête Prometheus de drift soutenu."""

import subprocess
import types

import pytest

from src.experiments import retrain_trigger


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _matrix_payload(values):
    return {
        "status": "success",
        "data": {"resultType": "matrix", "result": [{"metric": {"model": "m"}, "values": values}]},
    }


def test_query_drift_share_computes_fraction_of_ones(monkeypatch):
    payload = _matrix_payload([[1000, "1"], [1060, "1"], [1120, "0"], [1180, "1"]])
    monkeypatch.setattr(retrain_trigger.requests, "get", lambda *a, **k: _FakeResponse(payload))

    assert retrain_trigger.query_drift_share("http://prom:9090", "m") == pytest.approx(0.75)


def test_query_drift_share_zero_when_no_series(monkeypatch):
    payload = {"status": "success", "data": {"resultType": "matrix", "result": []}}
    monkeypatch.setattr(retrain_trigger.requests, "get", lambda *a, **k: _FakeResponse(payload))

    assert retrain_trigger.query_drift_share("http://prom:9090", "m") == 0.0


def test_is_drift_sustained_threshold():
    assert retrain_trigger.is_drift_sustained(0.85, min_share=0.8) is True
    assert retrain_trigger.is_drift_sustained(0.5, min_share=0.8) is False


def _git(repo, *args):
    subprocess.run(  # noqa: S603
        ["git", *args], cwd=repo, check=True, capture_output=True  # noqa: S607
    )


def _init_repo(repo):
    """``git init`` plus a local, CI-safe identity (no reliance on global git config)."""
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def _head(repo):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_latest_commit_touching_returns_none_without_history(tmp_path):
    _init_repo(tmp_path)
    assert retrain_trigger.latest_commit_touching(tmp_path, "data/data_drift") is None


def test_latest_commit_touching_finds_commit_on_path(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.txt").write_text("1")
    _git(tmp_path, "add", "data/x.txt")
    _git(tmp_path, "commit", "-m", "add data", "--author=t <t@t>")

    assert retrain_trigger.latest_commit_touching(tmp_path, "data/x.txt") == _head(tmp_path)


def test_latest_commit_touching_raises_when_not_a_git_repo(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        retrain_trigger.latest_commit_touching(tmp_path, "data/x.txt")


class _FakeRun:
    def __init__(self, tags):
        self.data = types.SimpleNamespace(tags=tags)


class _FakeSearchClient:
    """Double minimal : respecte le contrat réel de ``MlflowClient`` (``experiment_ids``,
    pas ``experiment_names`` -- ``search_runs`` ne les accepte pas, cf. ``retrain_trigger``),
    y compris la résolution ``get_experiment_by_name`` -> ``experiment_id`` préalable.
    """

    def __init__(self, runs, filter_string=None):
        self._runs = runs
        self.filter_string = filter_string
        self.search_calls = []

    def get_experiment_by_name(self, name):
        return types.SimpleNamespace(experiment_id="0", name=name)

    def search_runs(self, experiment_ids, filter_string=None, **kwargs):
        self.search_calls.append(filter_string)
        # Double "scopé" (cf. test du finding 3) : ne renvoie les runs que si
        # l'appelant a bien demandé le même filter_string -- un filtre absent
        # ou différent doit se comporter comme "aucun run ne correspond".
        if self.filter_string is not None and filter_string != self.filter_string:
            return []
        return self._runs


def test_latest_training_commit_reads_tag_from_most_recent_run():
    client = _FakeSearchClient([_FakeRun({"mlflow.source.git.commit": "abc123"})])
    assert retrain_trigger.latest_training_commit(client, "blood_cells-type") == "abc123"


def test_latest_training_commit_none_without_runs():
    client = _FakeSearchClient([])
    assert retrain_trigger.latest_training_commit(client, "blood_cells-type") is None


def test_latest_training_commit_scopes_search_with_run_name_filter():
    """Finding 3 (revue finale) : sans scoping, la recherche peut retomber sur un run
    sans rapport (grid sweep, autre backbone, run local) -- ``run_name`` doit ajouter un
    ``filter_string`` combinant le tag de nom de run et le statut FINISHED."""
    expected_filter = (
        "tags.mlflow.runName = 'retrain_on_drift' AND attributes.status = 'FINISHED'"
    )
    client = _FakeSearchClient(
        [_FakeRun({"mlflow.source.git.commit": "scoped123"})], filter_string=expected_filter
    )

    result = retrain_trigger.latest_training_commit(
        client, "blood_cells-type", run_name="retrain_on_drift"
    )

    assert result == "scoped123"
    assert client.search_calls == [expected_filter]


def test_latest_training_commit_without_run_name_does_not_filter():
    client = _FakeSearchClient([_FakeRun({"mlflow.source.git.commit": "abc123"})])

    retrain_trigger.latest_training_commit(client, "blood_cells-type")

    assert client.search_calls == [None]


def test_latest_training_commit_none_when_experiment_missing():
    client = _FakeSearchClient([_FakeRun({"mlflow.source.git.commit": "abc123"})])
    client.get_experiment_by_name = lambda name: None

    assert retrain_trigger.latest_training_commit(client, "does-not-exist") is None


def test_has_new_data_since_last_training_false_when_already_included(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "data_drift.txt").write_text("1")
    _git(tmp_path, "add", "data_drift.txt")
    _git(tmp_path, "commit", "-m", "a", "--author=t <t@t>")
    client = _FakeSearchClient([_FakeRun({"mlflow.source.git.commit": _head(tmp_path)})])

    result = retrain_trigger.has_new_data_since_last_training(
        tmp_path, "data_drift.txt", client, "exp"
    )
    assert result is False


def test_has_new_data_since_last_training_false_when_data_path_never_committed(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "other.txt").write_text("1")
    _git(tmp_path, "add", "other.txt")
    _git(tmp_path, "commit", "-m", "a", "--author=t <t@t>")
    (tmp_path / "data_drift.txt").write_text("1")  # jamais ajouté ni commité
    client = _FakeSearchClient([_FakeRun({"mlflow.source.git.commit": _head(tmp_path)})])

    result = retrain_trigger.has_new_data_since_last_training(
        tmp_path, "data_drift.txt", client, "exp"
    )
    assert result is False


def test_has_new_data_since_last_training_true_when_no_prior_run(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "data_drift.txt").write_text("1")
    _git(tmp_path, "add", "data_drift.txt")
    _git(tmp_path, "commit", "-m", "a", "--author=t <t@t>")
    client = _FakeSearchClient([])

    result = retrain_trigger.has_new_data_since_last_training(
        tmp_path, "data_drift.txt", client, "exp"
    )
    assert result is True


def test_has_new_data_since_last_training_true_when_new_commit_landed(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "data_drift.txt").write_text("1")
    _git(tmp_path, "add", "data_drift.txt")
    _git(tmp_path, "commit", "-m", "a", "--author=t <t@t>")
    old_commit = _head(tmp_path)
    (tmp_path / "data_drift.txt").write_text("2")
    _git(tmp_path, "add", "data_drift.txt")
    _git(tmp_path, "commit", "-m", "b", "--author=t <t@t>")
    client = _FakeSearchClient([_FakeRun({"mlflow.source.git.commit": old_commit})])

    result = retrain_trigger.has_new_data_since_last_training(
        tmp_path, "data_drift.txt", client, "exp"
    )
    assert result is True
