"""Tests pour le découpage ponctuel du socle munich (train/eval fixe, hardlinks)."""

import random
from pathlib import Path

import pytest

from src.helpers.munich_socle import build_munich_socle


def _noop_run(cmd, cwd):
    """Stub ``run`` for tests that don't care about the dvc/git calls."""


def test_build_munich_socle_splits_by_class_and_hardlinks(tmp_path):
    source = tmp_path / "source"
    for cls, n in [("BAS", 10), ("EOS", 10)]:
        (source / cls).mkdir(parents=True)
        for i in range(n):
            (source / cls / f"{cls}_{i:03d}.tiff").write_bytes(b"x")
    train_dest, eval_dest = tmp_path / "train", tmp_path / "test"

    build_munich_socle(
        eval_fraction=0.2,
        seed=0,
        source=source,
        train_dest=train_dest,
        eval_dest=eval_dest,
        run=_noop_run,
    )

    train_files = list(train_dest.glob("*.tiff"))
    eval_files = list(eval_dest.glob("*.tiff"))
    assert len(train_files) + len(eval_files) == 20
    assert 2 <= len(eval_files) <= 6  # ~20% de 20, les deux classes représentées
    assert {p.name for p in train_files}.isdisjoint({p.name for p in eval_files})
    sample = train_files[0]
    original = source / sample.name.split("_")[0] / sample.name
    assert sample.stat().st_ino == original.stat().st_ino  # hardlink, pas une copie


def test_build_munich_socle_refuses_to_run_twice(tmp_path):
    source = tmp_path / "source"
    (source / "BAS").mkdir(parents=True)
    (source / "BAS" / "BAS_000.tiff").write_bytes(b"x")
    train_dest, eval_dest = tmp_path / "train", tmp_path / "test"
    build_munich_socle(source=source, train_dest=train_dest, eval_dest=eval_dest, run=_noop_run)

    with pytest.raises(FileExistsError):
        build_munich_socle(source=source, train_dest=train_dest, eval_dest=eval_dest, run=_noop_run)


def test_build_munich_socle_is_deterministic_regardless_of_glob_order(tmp_path, monkeypatch):
    """Finding 3b: the split must not depend on filesystem iteration order."""
    source = tmp_path / "source"
    for cls, n in [("BAS", 10), ("EOS", 10)]:
        (source / cls).mkdir(parents=True)
        for i in range(n):
            (source / cls / f"{cls}_{i:03d}.tiff").write_bytes(b"x")

    real_glob = Path.glob

    def shuffled_glob(self, pattern):
        results = list(real_glob(self, pattern))
        random.Random(42).shuffle(results)  # noqa: S311
        return iter(results)

    monkeypatch.setattr(Path, "glob", shuffled_glob)

    train_dest_a, eval_dest_a = tmp_path / "train_a", tmp_path / "eval_a"
    build_munich_socle(
        eval_fraction=0.2,
        seed=0,
        source=source,
        train_dest=train_dest_a,
        eval_dest=eval_dest_a,
        run=_noop_run,
    )

    def reshuffled_glob(self, pattern):
        results = list(real_glob(self, pattern))
        random.Random(7).shuffle(results)  # noqa: S311
        return iter(results)

    monkeypatch.setattr(Path, "glob", reshuffled_glob)

    train_dest_b, eval_dest_b = tmp_path / "train_b", tmp_path / "eval_b"
    build_munich_socle(
        eval_fraction=0.2,
        seed=0,
        source=source,
        train_dest=train_dest_b,
        eval_dest=eval_dest_b,
        run=_noop_run,
    )

    assert {p.name for p in eval_dest_a.glob("*.tiff")} == {p.name for p in eval_dest_b.glob("*.tiff")}
    assert {p.name for p in train_dest_a.glob("*.tiff")} == {p.name for p in train_dest_b.glob("*.tiff")}


def test_build_munich_socle_commits_and_pushes_via_dvc_and_git(tmp_path):
    source = tmp_path / "source"
    (source / "BAS").mkdir(parents=True)
    for i in range(5):
        (source / "BAS" / f"BAS_{i:03d}.tiff").write_bytes(b"x")
    train_dest, eval_dest = tmp_path / "train", tmp_path / "test"
    repo_root = tmp_path / "repo"
    calls = []

    build_munich_socle(
        eval_fraction=0.2,
        seed=0,
        source=source,
        train_dest=train_dest,
        eval_dest=eval_dest,
        repo_root=repo_root,
        run=lambda cmd, cwd: calls.append((cmd, cwd)),
    )

    joined = [" ".join(c) for c, _ in calls]
    assert any(c.startswith("dvc add") for c in joined)
    assert any(c.startswith("git add") for c in joined)
    assert any(c.startswith("git commit") for c in joined)
    assert any(c.startswith("dvc push") for c in joined)
    assert any(c.startswith("git push") for c in joined)
    assert all(cwd == repo_root for _, cwd in calls)
    commit_call = next(c for c in calls if c[0][:2] == ["git", "commit"])
    assert str(train_dest) + ".dvc" in commit_call[0]
    assert str(eval_dest) + ".dvc" in commit_call[0]
