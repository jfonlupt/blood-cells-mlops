"""Tests pour la capture/réservation continue des images barcelone en dérive."""

import pandas as pd

from src.helpers import data_drift


def test_persist_barcelona_batch_splits_and_hardlinks(tmp_path, monkeypatch):
    src = tmp_path / "src" / "BA_0.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    df = pd.DataFrame({"path": [str(src)], "dataset": ["barcelona"]})
    monkeypatch.setattr(data_drift, "DRIFT_TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(data_drift, "DRIFT_EVAL_DIR", tmp_path / "eval")

    batch = data_drift.persist_barcelona_batch(df, train_fraction=0.8, seed=0)

    assert len(batch.train_paths) + len(batch.eval_paths) == 1
    linked = (tmp_path / "train" / "BA_0.jpg").exists() or (tmp_path / "eval" / "BA_0.jpg").exists()
    assert linked


def test_persist_barcelona_batch_ignores_non_barcelona_rows(tmp_path, monkeypatch):
    df = pd.DataFrame({"path": ["/x/MUC_0.jpg"], "dataset": ["munich"]})
    monkeypatch.setattr(data_drift, "DRIFT_TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(data_drift, "DRIFT_EVAL_DIR", tmp_path / "eval")

    batch = data_drift.persist_barcelona_batch(df)

    assert batch.train_paths == [] and batch.eval_paths == []


def test_persist_barcelona_batch_is_idempotent(tmp_path, monkeypatch):
    src = tmp_path / "src" / "BA_0.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    df = pd.DataFrame({"path": [str(src)], "dataset": ["barcelona"]})
    monkeypatch.setattr(data_drift, "DRIFT_TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(data_drift, "DRIFT_EVAL_DIR", tmp_path / "eval")

    first = data_drift.persist_barcelona_batch(df, seed=0)
    second = data_drift.persist_barcelona_batch(df, seed=0)

    assert len(first.train_paths) + len(first.eval_paths) == 1
    assert second.train_paths == [] and second.eval_paths == []


def test_persist_barcelona_batch_creates_both_dirs_even_when_only_one_is_populated(
    tmp_path, monkeypatch
):
    """Finding 2: early/small batches may only populate one side; both dirs
    must still exist afterwards so a later `dvc add train eval` never fails
    on a missing directory."""
    src = tmp_path / "src" / "BA_0.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    df = pd.DataFrame({"path": [str(src)], "dataset": ["barcelona"]})
    monkeypatch.setattr(data_drift, "DRIFT_TRAIN_DIR", tmp_path / "train")
    monkeypatch.setattr(data_drift, "DRIFT_EVAL_DIR", tmp_path / "eval")

    # train_fraction=1.0 forces every row into the train pool.
    batch = data_drift.persist_barcelona_batch(df, train_fraction=1.0, seed=0)

    assert batch.eval_paths == []
    assert len(batch.train_paths) == 1
    assert (tmp_path / "train").exists()
    assert (tmp_path / "eval").exists()  # created even though nothing landed here


def test_commit_and_push_batch_runs_dvc_and_git(tmp_path):
    calls = []
    batch = data_drift.CapturedBatch(train_paths=[tmp_path / "train" / "a.jpg"], eval_paths=[])

    data_drift.commit_and_push_batch(
        batch, repo_root=tmp_path, run=lambda cmd, cwd: calls.append((cmd, cwd))
    )

    joined = [" ".join(c) for c, _ in calls]
    assert any(c.startswith("dvc add") for c in joined)
    assert any(c.startswith("git add") for c in joined)
    assert any(c.startswith("git commit") for c in joined)
    assert any(c.startswith("dvc push") for c in joined)
    assert any(c.startswith("git push") for c in joined)


def test_commit_and_push_batch_scopes_git_commit_to_dvc_paths(tmp_path):
    """Finding 5: `git commit` must be scoped to the two .dvc paths, not a
    bare commit of whatever else happens to be staged in the index."""
    calls = []
    batch = data_drift.CapturedBatch(train_paths=[tmp_path / "train" / "a.jpg"], eval_paths=[])

    data_drift.commit_and_push_batch(
        batch, repo_root=tmp_path, run=lambda cmd, cwd: calls.append((cmd, cwd))
    )

    commit_call = next(cmd for cmd, _ in calls if cmd[:2] == ["git", "commit"])
    assert "--" in commit_call
    scoped_paths = commit_call[commit_call.index("--") + 1 :]
    assert scoped_paths == [
        str(data_drift.DRIFT_TRAIN_DIR) + ".dvc",
        str(data_drift.DRIFT_EVAL_DIR) + ".dvc",
    ]


def test_commit_and_push_batch_noop_on_empty_batch(tmp_path):
    calls = []
    batch = data_drift.CapturedBatch(train_paths=[], eval_paths=[])

    data_drift.commit_and_push_batch(
        batch, repo_root=tmp_path, run=lambda cmd, cwd: calls.append((cmd, cwd))
    )

    assert calls == []
