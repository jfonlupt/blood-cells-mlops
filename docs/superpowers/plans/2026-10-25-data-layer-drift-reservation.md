# Data Layer & Drift Reservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the drift-retraining feature a physical train/test data layout (fixed munich socle + growing data_drift eval holdout, via hardlinks) and wire image capture/reservation into the drift simulator.

**Architecture:** New `DATASET_PATHS` keys point at hardlinked subtrees under `data/train/`/`data/test/`, built once for munich and continuously for barcelona images seen by the simulator. `get_images`/`create_dataframe` need no changes — they already scan recursively and derive labels from filenames.

**Tech Stack:** Python (stdlib `os.link`, `subprocess`), pandas, scikit-learn (`train_test_split`, already a transitive dependency via existing `split_off_test` usage), `dvc`/`git` CLIs.

**Spec:** [docs/superpowers/specs/2026-10-25-data-drift-retraining-design.md](../specs/2026-10-25-data-drift-retraining-design.md)

## Global Constraints

- Hardlinks only, never copies, between the existing flat dataset directories and the new train/test views (`os.link`) — the whole point is zero extra disk space, confirmed free thanks to `.dvc/config`'s `cache.type = "hardlink,symlink"`.
- `DATASET_PATHS[MUNICH_KEY]`/`DATASET_PATHS[BARCELONA_KEY]` and their existing `.dvc` tracking stay untouched — this plan only adds new keys, never shrinks or redefines the existing ones (~12 configs/~14 notebooks depend on them, unrelated to this feature).
- The munich socle carve-out is one-time by design: running it twice must raise, never silently reshuffle which images are held out (a reshuffle would invalidate every past promotion decision).
- Default reservation ratio for barcelona images: 80% `train_pool` / 20% `eval_holdout` (configurable parameter, not a hardcoded literal scattered across call sites).
- Capturing an already-captured image is a no-op, not an error (the simulator may redraw the same barcelona image across separate runs).
- Git/DVC commit batching: one commit per simulator iteration (one `/evaluate` call), never per image.

## Review Focus

1. Re-running the munich socle carve-out a second time (even by accident) must not reshuffle already-assigned images — covered by Task 2's `FileExistsError` guard and its dedicated test.
2. An unmappable filename (class prefix absent from `MAPPING_DICT`) must not reach the capture path at all — this is already handled upstream by `get_images`'s existing `if label in MAPPING_DICT[dataset_name]` filter (`src/features/dataframe.py:62`), so no new filtering logic is needed here; flagged so it isn't accidentally re-implemented.
3. Persisting the same barcelona image twice across two separate simulator runs must not raise (`FileExistsError` from `os.link`) and must not double-count in the returned batch — covered by Task 3's idempotency test.
4. An empty capture batch (no barcelona rows in a given `/evaluate` call) must not trigger an empty `git commit`/`dvc push` cycle — covered by Task 4's no-op test.
5. `os.link` failing across a filesystem boundary (`OSError`, errno `EXDEV`) must propagate with its native message, never be silently caught and replaced by a space-costing copy — called out as an implementation note in Task 3 rather than tested (not practically simulable with a single-filesystem `tmp_path`).

---

### Task 1: New dataset keys (constants, config, drift training config)

**Files:**
- Modify: `src/constants.py`
- Modify: `src/config.py`
- Modify: `configs/densenet_data_drift.yaml`
- Modify: `src/helpers/data_drift.py` (one line, `generate_data_drift`'s `training_conf`)
- Test: `tests/test_constants.py`

**Interfaces:**
- Produces: `MUNICH_TRAIN_KEY = "munich_train"`, `MUNICH_EVAL_KEY = "munich_eval"`, `DRIFT_TRAIN_KEY = "data_drift_train"`, `DRIFT_EVAL_KEY = "data_drift_eval"`; `DATASET_PATHS` entries `DATA_PATH/"train"/"munich"`, `DATA_PATH/"test"/"munich"`, `DATA_PATH/"train"/"data_drift"`, `DATA_PATH/"test"/"data_drift"`; `MAPPING_DICT` entries reusing `MUNICH_TO_FINAL`/`BARCELONA_TO_FINAL`. The flat `DRIFT_KEY`/`"data_drift"` key (added in an earlier draft, never consumed anywhere except the YAML this task also edits) is removed.

- [ ] **Step 1: Write the failing test**

```python
def test_new_drift_dataset_keys_resolve_under_data_dir(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/app/data")
    try:
        mod = _reload_constants()
        assert mod.DATASET_PATHS["munich_train"] == Path("/app/data/train/munich")
        assert mod.DATASET_PATHS["munich_eval"] == Path("/app/data/test/munich")
        assert mod.DATASET_PATHS["data_drift_train"] == Path("/app/data/train/data_drift")
        assert mod.DATASET_PATHS["data_drift_eval"] == Path("/app/data/test/data_drift")
        assert "data_drift" not in mod.DATASET_PATHS
        assert mod.DATASET_PATHS["munich"] == Path("/app/data/PKG-AML-Cytomorphology_LMU")
    finally:
        monkeypatch.delenv("DATA_DIR", raising=False)
        _reload_constants()
```

Add this next to the existing `test_dataset_paths_follow_data_dir_env_var` in `tests/test_constants.py`, reusing its `_reload_constants` helper.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_constants.py -v`
Expected: FAIL (`KeyError`/`AssertionError` — new keys absent)

- [ ] **Step 3: Implement the new keys in `src/constants.py`**

Add `MUNICH_TRAIN_KEY`, `MUNICH_EVAL_KEY`, `DRIFT_TRAIN_KEY`, `DRIFT_EVAL_KEY`. Replace the flat `DRIFT_KEY` and its `DATASET_PATHS`/`MAPPING_DICT` entries with the four new ones. Update `DATASETS_KEYS` to `[BARCELONA_KEY, MUNICH_KEY, MUNICH_TRAIN_KEY, MUNICH_EVAL_KEY, DRIFT_TRAIN_KEY, DRIFT_EVAL_KEY]`.

- [ ] **Step 4: Update `DataConfig.datasets` in `src/config.py`**

Change the `Literal[...]` on `DataConfig.datasets` (currently `Literal["barcelona", "munich", "data_drift"]`) to `Literal["barcelona", "munich", "munich_train", "munich_eval", "data_drift_train", "data_drift_eval"]`.

- [ ] **Step 5: Update `configs/densenet_data_drift.yaml`**

Change `data.datasets: [munich, data_drift]` to `data.datasets: [munich_train, data_drift_train]` — training must never read from the `*_eval` keys.

- [ ] **Step 6: Update `generate_data_drift` in `src/helpers/data_drift.py`**

Change `training_conf = DataConfig(datasets=["munich"], stratify="type")` to `datasets=["munich_eval"]`. The simulated baseline traffic should be drawn from images structurally guaranteed to never have been used to train any champion (this is exactly what the eval socle is for).

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/test_constants.py -v`
Expected: PASS

- [ ] **Step 8: Run the config test suite to confirm the Literal change is compatible**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/constants.py src/config.py configs/densenet_data_drift.yaml src/helpers/data_drift.py tests/test_constants.py
git commit -m "feat: add munich_train/munich_eval/data_drift_train/data_drift_eval dataset keys"
```

---

### Task 2: One-time munich socle carve-out

**Files:**
- Create: `src/helpers/munich_socle.py`
- Test: `tests/helpers/test_munich_socle.py`

**Interfaces:**
- Consumes: `DATASET_PATHS[MUNICH_KEY]`, `DATASET_PATHS[MUNICH_TRAIN_KEY]`, `DATASET_PATHS[MUNICH_EVAL_KEY]` (Task 1).
- Produces:
```python
def build_munich_socle(
    eval_fraction: float = 0.2,
    seed: int = 0,
    source: Path | None = None,
    train_dest: Path | None = None,
    eval_dest: Path | None = None,
) -> None: ...
```
Defaults resolve to `DATASET_PATHS[MUNICH_KEY]`/`DATASET_PATHS[MUNICH_TRAIN_KEY]`/`DATASET_PATHS[MUNICH_EVAL_KEY]`. Raises `FileExistsError` immediately (before linking anything) if `train_dest` or `eval_dest` already exists and is non-empty.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_munich_socle_splits_by_class_and_hardlinks(tmp_path):
    source = tmp_path / "source"
    for cls, n in [("BAS", 10), ("EOS", 10)]:
        (source / cls).mkdir(parents=True)
        for i in range(n):
            (source / cls / f"{cls}_{i:03d}.tiff").write_bytes(b"x")
    train_dest, eval_dest = tmp_path / "train", tmp_path / "test"

    build_munich_socle(
        eval_fraction=0.2, seed=0, source=source, train_dest=train_dest, eval_dest=eval_dest
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
    build_munich_socle(source=source, train_dest=train_dest, eval_dest=eval_dest)

    with pytest.raises(FileExistsError):
        build_munich_socle(source=source, train_dest=train_dest, eval_dest=eval_dest)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/helpers/test_munich_socle.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `build_munich_socle` in `src/helpers/munich_socle.py`**

Raise `FileExistsError` first if either destination exists and contains any file. Recursively glob `source` for `*.jpg`/`*.png`/`*.tiff` (same extensions as `get_images`), derive the class from `p.stem.split("_")[0]` (same convention as `get_images`, `src/features/dataframe.py:58`). Stratified split per class via `sklearn.model_selection.train_test_split(..., test_size=eval_fraction, random_state=seed)`, consistent with the project's existing `split_off_test`. `dest_dir.mkdir(parents=True, exist_ok=True)` then `os.link(src, dest_dir / src.name)` for each file — flat destination, no nested subdirs needed (`get_images`'s recursive glob doesn't care about depth).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/helpers/test_munich_socle.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/helpers/munich_socle.py tests/helpers/test_munich_socle.py
git commit -m "feat: add one-time munich socle carve-out"
```

---

### Task 3: Continuous capture/reservation for barcelona images

**Files:**
- Modify: `src/helpers/data_drift.py`
- Test: `tests/helpers/test_data_drift.py` (new)

**Interfaces:**
- Consumes: `DATASET_PATHS[DRIFT_TRAIN_KEY]`, `DATASET_PATHS[DRIFT_EVAL_KEY]` (Task 1).
- Produces:
```python
@dataclass
class CapturedBatch:
    train_paths: list[Path]
    eval_paths: list[Path]

def persist_barcelona_batch(
    df: pd.DataFrame, train_fraction: float = 0.8, seed: int | None = None
) -> CapturedBatch: ...
```
`df` must have `path`/`dataset` columns (as produced by `create_dataframe`). Only rows with `dataset == "barcelona"` are considered; others are ignored. The returned `CapturedBatch` holds only the paths newly linked by *this* call — a path already present at its destination from an earlier call is skipped and does not appear in either list. Module-level `DRIFT_TRAIN_DIR`/`DRIFT_EVAL_DIR` attributes (defaulting to `DATASET_PATHS[DRIFT_TRAIN_KEY]`/`DATASET_PATHS[DRIFT_EVAL_KEY]`) must exist so tests can `monkeypatch.setattr(data_drift, "DRIFT_TRAIN_DIR", ...)`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/helpers/test_data_drift.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `CapturedBatch` and `persist_barcelona_batch` in `src/helpers/data_drift.py`**

Add module-level `DRIFT_TRAIN_DIR = DATASET_PATHS[DRIFT_TRAIN_KEY]` / `DRIFT_EVAL_DIR = DATASET_PATHS[DRIFT_EVAL_KEY]`. For each row where `dataset == "barcelona"`: skip (don't add to either list) if a same-named file already exists in `DRIFT_TRAIN_DIR` or `DRIFT_EVAL_DIR`; otherwise draw with `random.Random(seed)` (one RNG instance per call, not reseeded per row) — `train_fraction` probability of `train_pool`; `mkdir(parents=True, exist_ok=True)` the chosen destination, then `os.link(path, dest_dir / Path(path).name)`. Let any `OSError` (including `EXDEV` across filesystems) propagate unmodified — do not wrap `os.link` in a broad `except`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/helpers/test_data_drift.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/helpers/data_drift.py tests/helpers/test_data_drift.py
git commit -m "feat: capture and reserve barcelona images into train/eval holdout"
```

---

### Task 4: Batched git/dvc commit, wired into the simulator loop

**Files:**
- Modify: `src/helpers/data_drift.py`
- Test: `tests/helpers/test_data_drift.py` (extend)

**Interfaces:**
- Consumes: `CapturedBatch` (Task 3).
- Produces:
```python
def commit_and_push_batch(
    batch: CapturedBatch,
    repo_root: Path = BASE_DIR,
    run: Callable[[list[str], Path], None] = _run_git,
) -> None: ...
```
No subprocess call at all (not even `dvc add`) if `batch.train_paths` and `batch.eval_paths` are both empty.

- [ ] **Step 1: Write the failing tests**

```python
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


def test_commit_and_push_batch_noop_on_empty_batch(tmp_path):
    calls = []
    batch = data_drift.CapturedBatch(train_paths=[], eval_paths=[])

    data_drift.commit_and_push_batch(
        batch, repo_root=tmp_path, run=lambda cmd, cwd: calls.append((cmd, cwd))
    )

    assert calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/helpers/test_data_drift.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `commit_and_push_batch` and `_run_git`**

Early-return (no calls at all) if both path lists are empty. Otherwise, in order: `run(["dvc", "add", str(DRIFT_TRAIN_DIR), str(DRIFT_EVAL_DIR)], repo_root)`, `run(["git", "add", str(DRIFT_TRAIN_DIR) + ".dvc", str(DRIFT_EVAL_DIR) + ".dvc"], repo_root)`, `run(["git", "commit", "-m", f"data_drift: +{len(batch.train_paths)} train / +{len(batch.eval_paths)} eval"], repo_root)`, `run(["dvc", "push"], repo_root)`, `run(["git", "push"], repo_root)`. `_run_git(cmd, cwd)` (the default `run`) is `subprocess.run(cmd, cwd=cwd, check=True)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/helpers/test_data_drift.py -v`
Expected: PASS

- [ ] **Step 5: Wire into `main()`'s loop**

Replace the stubbed TODO block. Fix the pre-existing stale filter in the comment (`df["dataset"] == "data_drift"` never matches — `create_dataframe` tags origin rows `"barcelona"`/`"munich"`, never `"data_drift"`): use `barcelona_rows = pred_df[pred_df["dataset"] == "barcelona"]`, then `batch = persist_barcelona_batch(barcelona_rows)` followed by `commit_and_push_batch(batch)`, once per `/evaluate` call.

- [ ] **Step 6: Run the full helpers test suite**

Run: `uv run pytest tests/helpers/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/helpers/data_drift.py tests/helpers/test_data_drift.py
git commit -m "feat: batch-commit captured drift images via dvc/git"
```
