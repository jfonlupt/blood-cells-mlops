"""Test d'intégration de bout en bout : socle munich -> capture barcelone -> commit dvc/git.

Contrairement aux tests unitaires de ``test_munich_socle.py``/``test_data_drift.py``
(qui injectent un ``run`` factice), ce test fait tourner de vraies commandes
``git``/``dvc`` contre un dépôt jetable dans ``tmp_path``, pour vérifier que les
trois tâches du plan s'enchaînent correctement une fois assemblées :
carve-out du socle munich -> capture d'un batch barcelone -> commit/push.

Il n'exécute jamais de vrai ``dvc push``/``git push`` (pas de remote configuré
dans ce dépôt jetable) : ces deux commandes sont interceptées en no-op par le
``run`` injecté, qui laisse passer toutes les autres commandes vers un vrai
``subprocess.run``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.helpers import data_drift
from src.helpers.munich_socle import build_munich_socle

PROJECT_GITIGNORE = Path(__file__).resolve().parents[2] / ".gitignore"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("dvc") is None,
    reason="git/dvc CLI not available on PATH",
)


def _run_cmd(cmd: list[str], cwd: Path, *, capture: bool = False) -> subprocess.CompletedProcess:
    """Single choke point for real subprocess calls in this test.

    Command names (``git``/``dvc``) are fixed literals controlled by this
    test file, never user input, so a partial executable path is safe here
    -- same justification as ``_run_git`` in ``src/helpers/data_drift.py``.
    """
    return subprocess.run(  # noqa: S603
        cmd, cwd=cwd, check=True, capture_output=capture, text=capture
    )


def _real_run(cmd: list[str], cwd: Path) -> None:
    """Injected ``run``: real subprocess for everything except push.

    There is no remote configured for this throwaway repo, so `dvc push`/
    `git push` are no-ops here -- this test's job is to verify local
    carve-out + capture + commit wiring and git tracking, not networking.
    """
    if cmd[:2] in (["dvc", "push"], ["git", "push"]):
        return
    _run_cmd(cmd, cwd)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    _run_cmd(["git", "init", "-q"], repo)
    _run_cmd(["git", "config", "user.email", "test@example.com"], repo)
    _run_cmd(["git", "config", "user.name", "Test"], repo)
    # Use the project's real .gitignore so this test also exercises finding 1
    # (nested data/**/*.dvc files must not be ignored) end-to-end.
    shutil.copy(PROJECT_GITIGNORE, repo / ".gitignore")
    _run_cmd(["dvc", "init", "-q"], repo)
    _run_cmd(["git", "add", ".gitignore", ".dvc"], repo)
    _run_cmd(["git", "commit", "-q", "-m", "init throwaway repo"], repo)


def test_socle_then_capture_pipeline_tracks_dvc_files_in_git(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _init_repo(repo)

    # --- Task 2: carve out the munich socle -------------------------------
    munich_source = tmp_path / "munich_source"
    for cls, n in [("BAS", 10), ("EOS", 10)]:
        (munich_source / cls).mkdir(parents=True)
        for i in range(n):
            (munich_source / cls / f"{cls}_{i:03d}.tiff").write_bytes(b"x")

    munich_train_dest = repo / "data" / "train" / "munich"
    munich_eval_dest = repo / "data" / "test" / "munich"

    build_munich_socle(
        eval_fraction=0.2,
        seed=0,
        source=munich_source,
        train_dest=munich_train_dest,
        eval_dest=munich_eval_dest,
        repo_root=repo,
        run=_real_run,
    )

    assert any(munich_train_dest.iterdir())
    assert any(munich_eval_dest.iterdir())

    # --- Task 3+4: capture a barcelona batch and commit it ----------------
    barcelona_source = tmp_path / "barcelona_source"
    barcelona_source.mkdir()
    barcelona_img = barcelona_source / "BA_0.jpg"
    barcelona_img.write_bytes(b"x")

    drift_train_dir = repo / "data" / "train" / "data_drift"
    drift_eval_dir = repo / "data" / "test" / "data_drift"
    monkeypatch.setattr(data_drift, "DRIFT_TRAIN_DIR", drift_train_dir)
    monkeypatch.setattr(data_drift, "DRIFT_EVAL_DIR", drift_eval_dir)

    df = pd.DataFrame({"path": [str(barcelona_img)], "dataset": ["barcelona"]})
    # train_fraction=1.0: forces the single captured image into the train
    # pool only, exercising finding 2 (dvc add must still succeed even
    # though the eval side has nothing but an empty, newly-created dir).
    batch = data_drift.persist_barcelona_batch(df, train_fraction=1.0, seed=0)
    assert len(batch.train_paths) == 1
    assert batch.eval_paths == []
    assert drift_eval_dir.exists()  # created up front even though unused

    data_drift.commit_and_push_batch(batch, repo_root=repo, run=_real_run)

    # --- Assertions: everything a teammate re-cloning would need is committed ---
    tracked = _run_cmd(["git", "ls-files"], repo, capture=True).stdout.splitlines()

    assert "data/train/munich.dvc" in tracked
    assert "data/test/munich.dvc" in tracked
    assert "data/train/data_drift.dvc" in tracked
    assert "data/test/data_drift.dvc" in tracked

    # Working tree is clean: nothing left dangling, uncommitted.
    status = _run_cmd(["git", "status", "--porcelain"], repo, capture=True).stdout
    assert status.strip() == ""
