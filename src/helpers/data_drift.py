"""Helper methods for data."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import requests

if TYPE_CHECKING:
    from collections.abc import Callable

from src.config import DataConfig
from src.constants import BASE_DIR, DATASET_PATHS, DRIFT_EVAL_KEY, DRIFT_TRAIN_KEY
from src.experiments.runner import seed_everything
from src.features.dataframe import create_dataframe
from src.tasks import TASKS
from src.training.components import split_off_test

DRIFT_TRAIN_DIR = DATASET_PATHS[DRIFT_TRAIN_KEY]
DRIFT_EVAL_DIR = DATASET_PATHS[DRIFT_EVAL_KEY]

# Branche jetable sur laquelle ce script est censé committer/pousser (cf.
# commit_and_push_batch) : permet de tout supprimer et reproduire la
# simulation de drift depuis zéro sans jamais polluer romain/main.
DRIFT_SIM_BRANCH = "data-drift-sim"


@dataclass
class CapturedBatch:
    train_paths: list[Path]
    eval_paths: list[Path]


def persist_barcelona_batch(
    df: pd.DataFrame, train_fraction: float = 0.8, seed: int | None = None
) -> CapturedBatch:
    """Capture barcelona rows of ``df`` into the train pool or the eval holdout.

    Each barcelona image is assigned, the first time it is seen, to either
    ``DRIFT_TRAIN_DIR`` or ``DRIFT_EVAL_DIR`` by hardlink, decided by a single
    RNG draw per row (``train_fraction`` probability of the train pool). A
    row whose image already exists at either destination (from an earlier
    call) is skipped and is not reported in the returned batch, so calling
    this repeatedly on overlapping batches is safe and never double-counts.

    Args:
        df: Dataframe with at least ``path``/``dataset`` columns, as produced
            by ``create_dataframe``. Only rows with ``dataset == "barcelona"``
            are considered; other rows are ignored.
        train_fraction: Probability that a newly-seen image is assigned to
            the train pool rather than the eval holdout.
        seed: Seed for the RNG driving the train/eval draw.

    Returns:
        A ``CapturedBatch`` listing only the paths newly linked by this call.
    """
    rng = random.Random(seed)  # noqa: S311
    train_paths: list[Path] = []
    eval_paths: list[Path] = []

    # Create both directories up front (even if this call captures nothing,
    # or captures only into one of the two) so that a later `dvc add
    # DRIFT_TRAIN_DIR DRIFT_EVAL_DIR` in commit_and_push_batch never fails
    # because one side doesn't exist yet -- early batches can easily draw
    # only train or only eval.
    DRIFT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    for path in df.loc[df["dataset"] == "barcelona", "path"]:
        name = Path(path).name
        if (DRIFT_TRAIN_DIR / name).exists() or (DRIFT_EVAL_DIR / name).exists():
            continue

        if rng.random() < train_fraction:
            dest_dir, captured = DRIFT_TRAIN_DIR, train_paths
        else:
            dest_dir, captured = DRIFT_EVAL_DIR, eval_paths

        dest = dest_dir / name
        os.link(path, dest)
        captured.append(dest)

    return CapturedBatch(train_paths=train_paths, eval_paths=eval_paths)


def _run_git(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)  # noqa: S603


def commit_and_push_batch(
    batch: CapturedBatch,
    repo_root: Path = BASE_DIR,
    run: Callable[[list[str], Path], None] = _run_git,
) -> None:
    """Commit and push newly captured drift images via dvc and git.

    Operates on ``DRIFT_TRAIN_DIR``/``DRIFT_EVAL_DIR`` as whole directories
    (not on individual file paths from ``batch``). Does nothing at all,
    not even ``dvc add``, when ``batch`` is empty.

    Args:
        batch: The ``CapturedBatch`` just produced by ``persist_barcelona_batch``.
        repo_root: Working directory for the dvc/git commands.
        run: Injectable command runner, defaulting to a real subprocess call.
    """
    if not batch.train_paths and not batch.eval_paths:
        return

    dvc_paths = [str(DRIFT_TRAIN_DIR) + ".dvc", str(DRIFT_EVAL_DIR) + ".dvc"]

    run(["dvc", "add", str(DRIFT_TRAIN_DIR), str(DRIFT_EVAL_DIR)], repo_root)
    run(["git", "add", *dvc_paths], repo_root)
    run(
        [
            "git",
            "commit",
            "-m",
            f"data_drift: +{len(batch.train_paths)} train / +{len(batch.eval_paths)} eval",
            "--",
            *dvc_paths,
        ],
        repo_root,
    )
    run(["dvc", "push"], repo_root)
    run(["git", "push"], repo_root)


def generate_data_drift(seed: int = 0, batch_size: int = 50):
    seed_everything(seed)
    task = TASKS["type"]
    training_conf = DataConfig(datasets=["munich_eval"], stratify="type")
    training_set = create_dataframe(training_conf, task)
    _, test_df = split_off_test(training_set, training_conf, seed)
    if len(test_df) < batch_size:
        raise ValueError(
            f"batch size ({batch_size}) must be smaller than the test set ({len(test_df)})"
        )

    other_conf = DataConfig(datasets=["barcelona"], stratify="type")
    other_df = create_dataframe(other_conf, task).sample(frac=1)

    data_drifted = False
    n, start = 0, 0
    while not data_drifted and n < batch_size:
        n += 1
        df = pd.concat([test_df.sample(batch_size - n), other_df.iloc[start : start + n]])
        start += n
        yield df[["path", "dataset", "type"]]


def _current_branch(repo_root: Path = BASE_DIR) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _confirm_branch_or_exit(repo_root: Path = BASE_DIR) -> None:
    """Prompt for confirmation when not on ``DRIFT_SIM_BRANCH``.

    ``commit_and_push_batch`` commits and pushes on whatever branch is
    currently checked out ; running this script outside of
    ``DRIFT_SIM_BRANCH`` would otherwise silently push simulated-drift data
    onto romain/main.
    """
    branch = _current_branch(repo_root)
    if branch == DRIFT_SIM_BRANCH:
        return
    answer = input(
        f"Branche courante : {branch!r} (pas {DRIFT_SIM_BRANCH!r}). "
        "Les commits de simulation de drift vont atterrir ici. Continuer ? [y/N] "
    )
    if answer.strip().lower() not in ("y", "yes", "o", "oui"):
        raise SystemExit(f"Annulé : `git checkout {DRIFT_SIM_BRANCH}` avant de relancer.")


def _post_evaluate(
    url: str,
    token: str,
    df: pd.DataFrame,
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
) -> requests.Response:
    """POST ``/evaluate``, avec retry sur erreur serveur transitoire (5xx).

    ``BloodCellsService`` délègue ``/evaluate`` à ``BloodCellModelService`` via
    le client HTTP interne de BentoML, dont le pool de connexions keep-alive
    peut réutiliser une connexion déjà fermée côté serveur
    (``BrokenPipeError``/``ServerDisconnectedError``, remonté ici comme un
    500) -- panne d'infra sporadique, sans rapport avec les données envoyées
    (constaté le 2026-10-26, cf. logs ``blood_cells_server``). Ne retente que
    sur 5xx ; une erreur 4xx est levée immédiatement.
    """
    for attempt in range(1, max_attempts + 1):
        with ExitStack() as stack:
            resp = requests.post(
                url=url + "/evaluate",
                headers={"Authorization": f"Bearer {token}"},
                files=[
                    ("model", (None, "densenet121-type")),
                    ("labels", (None, json.dumps(df["type"].to_list()))),
                ]
                + [
                    (
                        "images",
                        (
                            Path(p).name,
                            stack.enter_context(Path.open(p, "rb")),
                        ),
                    )
                    for p in df["path"]
                ],
                timeout=60,
            )
        if resp.status_code < 500 or attempt == max_attempts:
            resp.raise_for_status()
            return resp
        print(f"/evaluate: {resp.status_code}, retry {attempt}/{max_attempts - 1}...")
        time.sleep(backoff_seconds)
    raise AssertionError("unreachable: last attempt always returns or raises")


def main():
    _confirm_branch_or_exit()

    url = "http://127.0.0.1:8000"
    resp = requests.post(
        url=url + "/login",
        headers={"Content-Type": "application/json"},
        json={
            "credentials": {
                "username": os.environ.get("BENTO_USER"),
                "password": os.environ.get("BENTO_PASSWORD"),
            }
        },
        timeout=15,
    )
    resp.raise_for_status()
    token = resp.json()["token"]
    batch_size = 50

    for i, df in enumerate(generate_data_drift(batch_size=batch_size)):
        # time.sleep(random.randint(1, 60))
        # n_pred = random.randint(1, len(df))
        # pred_df = df.sample(n_pred)
        # with ExitStack() as stack:
        #     resp = requests.post(
        #         url=url + "/predict",
        #         headers={"Authorization": f"Bearer {token}"},
        #         data={"model": "densenet121-type"},
        #         files=[
        #             (
        #                 "images",
        #                 (
        #                     Path(p).name,
        #                     stack.enter_context(Path.open(p, "rb")),
        #                 ),
        #             )
        #             for p in pred_df["path"]
        #         ],
        #         timeout=15,
        #     )
        # resp.raise_for_status()

        # time.sleep(random.randint(1, 60))

        barcelona_rows = df[df["dataset"] == "barcelona"]
        batch = persist_barcelona_batch(barcelona_rows)
        commit_and_push_batch(batch)

        print(f"evaluation {i}/{batch_size}")
        start_time = time.time()
        resp = _post_evaluate(url, token, df)
        print(f"evaluate took {time.time() - start_time:.2f}s")
        metrics = resp.json()["metrics"]
        print(f"Drifted columns : {metrics[4]['value']}")
        print(f"Embeddings : {metrics[-1]['value']}")


if __name__ == "__main__":
    main()
