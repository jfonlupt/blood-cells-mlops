# Airflow Retrain Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Airflow DAG that polls for sustained, non-redundant drift, triggers a retraining run with a fresh `GIT_COMMIT`, and promotes the result only if it beats the champion on both held-out evaluation sets.

**Architecture:** All decision logic (Prometheus polling, the git-commit redundancy guard, the promotion decision, the switchable metric registry) lives in plain, pytest-testable `src/experiments/` modules with no Airflow import. `dags/retrain_on_drift.py` is a thin wiring layer over those modules — it is not unit-tested in this repo's suite (Airflow isn't a dependency of this project's venv) and is verified manually in the Airflow UI as the final task's own step.

**Tech Stack:** Python (stdlib `subprocess`, `requests`), MLflow client API, Airflow 3.3.2 (`DockerOperator`, `@task`/`@task.branch`, `@dag`) for the DAG file only.

**Spec:** [docs/superpowers/specs/2026-10-25-data-drift-retraining-design.md](../specs/2026-10-25-data-drift-retraining-design.md)

**Depends on:** [2026-10-25-data-layer-drift-reservation.md](2026-10-25-data-layer-drift-reservation.md) (the `*_train`/`*_eval` dataset keys Task 5 evaluates against) and [2026-10-25-server-drift-signals.md](2026-10-25-server-drift-signals.md) (the `model_drift_detected` gauge Task 2 polls).

## Global Constraints

- `GIT_COMMIT` must be refreshed via a **runtime environment override** on the training container (`DockerOperator(environment={"GIT_COMMIT": ...})`), never by rebuilding the `blood_cells_training` image on each cycle — the image is only ever built by `make build-pipeline`, separately from running it.
- The redundancy guard compares git commits (`mlflow.source.git.commit` tag on the most recent MLflow run vs. the latest commit touching the data path) — no Airflow Variable or other separate state store.
- `max_active_runs=1` on the DAG, on top of the git-commit guard — the two are complementary (concurrent vs. consecutive-redundant), neither replaces the other.
- The challenger is resolved by `run_id`, never via the existing `"challenger"` MLflow alias (`register_model` in `tracking.py` sets that alias automatically, unconditionally, on every run, using its own metric/split — unrelated to this pipeline's promotion decision, and must not be disabled or relied upon here).
- The promotion metric must be switchable via a plain parameter/registry lookup, not hardcoded inline — `f1` is the current default, but swapping to a different metric later must not touch the decision logic itself.
- `src/experiments/promote.py`/`promote_cli.py` is an unrelated tool (epoch-extension of leaderboard runs) — nothing in this plan touches it, despite the name overlap with "promotion."

## Review Focus

1. A training run triggered while the repo's `data/train/data_drift.dvc` path has *not* changed since the last training run must be skipped, even if the drift gauge is still firing — covered by Task 3's `has_new_data_since_last_training` "false when already included" test. (Corrected post-implementation: the data-layer plan's simulator commits `data/train/data_drift.dvc`/`data/test/data_drift.dvc`, not a bare `data/data_drift` path — the image folders themselves are gitignored. This document originally said `data/data_drift`, predating that plan's later train/test physical-split decision; Task 6's final review caught the mismatch.)
2. A challenger that improves on the recent/drifted eval set but regresses beyond the margin on the fixed munich socle must *not* be promoted — covered by Task 4's "false when regression beyond margin" test.
3. A challenger that is better on both eval sets but the champion has no prior MLflow run at all (first-ever drift retrain) must still be evaluated and promoted correctly — covered by Task 3's "true when no prior run" test (the no-prior-run path is treated as "new data" so training proceeds); Task 5's `promote_if_better` must also tolerate `get_champion_metric`-style "no champion yet" (reuse the existing `tracking.get_champion_metric` pattern for this, don't reinvent it).
4. An empty or malformed Prometheus response (no matching series, or a request error) must resolve to "no sustained drift," never crash the DAG or be misread as "drift detected" — covered by Task 2's "zero when no series" test; a `requests` exception is deliberately left unhandled (propagates to Airflow's own task-retry/alerting, which is the correct place for infra failures to surface, not a silently-swallowed `False`).
5. `evaluate_on_eval_sets`/`promote_if_better` running against a champion and challenger trained with *different* `task`/model architectures must fail loudly rather than silently compare incomparable metrics — not covered by a dedicated test below; flagged as a gap for the implementer, since both models in this pipeline are always produced from the same `configs/densenet_data_drift.yaml`, making it unreachable in practice today, but worth a guard (`assert champion_task.name == challenger_task.name`) if this pipeline is ever reused for another model family.

---

### Task 1: Expose the MLflow run id from the training entrypoint

**Files:**
- Modify: `src/experiments/artifacts.py` (`RunResult`)
- Modify: `src/experiments/tracking.py` (`log_run`)
- Modify: `src/experiments/run.py` (CLI `main`)
- Test: `tests/experiments/test_tracking.py` (extend)

**Interfaces:**
- Produces: `RunResult.mlflow_run_id: str | None = None` (new field, default `None`); `log_run` sets `result.mlflow_run_id` to the active run's id before returning (no-op, i.e. stays `None`, when tracking is disabled); `run.py`'s `main()` prints `f"MLflow run id : {result.mlflow_run_id}"` as its own output line — this exact line is the contract Task 6's `DockerOperator`/XCom wiring parses out of the container's stdout.

- [ ] **Step 1: Read the existing `_FakeMlflow` double before writing the test**

Open `tests/experiments/test_tracking.py` and check whether `_FakeMlflow.start_run`'s context manager already exposes a `run_id`/`.info.run_id`-like attribute. Reuse whatever it already exposes; only extend `_FakeMlflow` if it doesn't, rather than adding a second, parallel mechanism.

- [ ] **Step 2: Write the failing test**

```python
def test_log_run_sets_mlflow_run_id_on_result(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setattr(tracking, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    result = _result()

    tracking.log_run(result, Path("/tmp/run"), torch.nn.Linear(1, 1), np.zeros((1, 1), dtype="float32"))

    assert result.mlflow_run_id is not None
    assert isinstance(result.mlflow_run_id, str)


def test_log_run_leaves_mlflow_run_id_none_when_tracking_disabled(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    result = _result()

    tracking.log_run(result, Path("/tmp/run"), torch.nn.Linear(1, 1), np.zeros((1, 1), dtype="float32"))

    assert result.mlflow_run_id is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/experiments/test_tracking.py -v`
Expected: FAIL (`AttributeError: 'RunResult' object has no attribute 'mlflow_run_id'`)

- [ ] **Step 4: Add the field to `RunResult`**

In `src/experiments/artifacts.py`, add `mlflow_run_id: str | None = None` to the `RunResult` dataclass.

- [ ] **Step 5: Set it inside `log_run`**

In `src/experiments/tracking.py`, change `with mlflow.start_run(run_name=result.name, nested=...):` to `with mlflow.start_run(run_name=result.name, nested=...) as active_run:`, and add `result.mlflow_run_id = active_run.info.run_id` as the first line inside the block.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/test_tracking.py -v`
Expected: PASS

- [ ] **Step 7: Print it from the CLI**

In `src/experiments/run.py`'s `main()`, add `print(f"MLflow run id : {result.mlflow_run_id}")` after the existing `print(f"Métriques test : {result.test_metrics}")`.

- [ ] **Step 8: Run the full experiments test suite**

Run: `uv run pytest tests/experiments/ -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/experiments/artifacts.py src/experiments/tracking.py src/experiments/run.py tests/experiments/test_tracking.py
git commit -m "feat: expose mlflow run id from the training entrypoint"
```

---

### Task 2: Prometheus sustained-drift query

**Files:**
- Create: `src/experiments/retrain_trigger.py`
- Test: `tests/experiments/test_retrain_trigger.py` (new)

**Interfaces:**
- Produces:
```python
def query_drift_share(
    prometheus_url: str, model: str, window_minutes: int = 15,
    step_seconds: int = 60, now: float | None = None,
) -> float: ...

def is_drift_sustained(share: float, min_share: float = 0.8) -> bool: ...
```
`now` defaults to `time.time()`. `query_drift_share` returns `0.0` when Prometheus reports no matching series (no data is treated as "no drift signal," never as an error).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/experiments/test_retrain_trigger.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `query_drift_share` and `is_drift_sustained` in `src/experiments/retrain_trigger.py`**

`query_drift_share` calls `requests.get(f"{prometheus_url}/api/v1/query_range", params={"query": f'model_drift_detected{{model="{model}"}}', "start": now - window_minutes * 60, "end": now, "step": step_seconds}, timeout=15)`, `resp.raise_for_status()`, reads `payload["data"]["result"]`; if empty, return `0.0`; otherwise flatten every series' `values` and return `statistics.fmean(float(v) for _, v in values)`. `is_drift_sustained(share, min_share=0.8)` is `share >= min_share`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/test_retrain_trigger.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/retrain_trigger.py tests/experiments/test_retrain_trigger.py
git commit -m "feat: add Prometheus sustained-drift query helper"
```

---

### Task 3: Git-commit-based redundant-retraining guard

**Files:**
- Modify: `src/experiments/retrain_trigger.py`
- Test: `tests/experiments/test_retrain_trigger.py` (extend)

**Interfaces:**
- Consumes: the `mlflow.source.git.commit` tag already written by `tracking._git_tags()`/`log_run` (no change needed there — it's already written today whenever `GIT_COMMIT` is set in the environment).
- Produces:
```python
def latest_commit_touching(repo_dir: Path, relative_path: str) -> str | None: ...

def latest_training_commit(client, experiment_name: str) -> str | None: ...

def has_new_data_since_last_training(
    repo_dir: Path, relative_path: str, client, experiment_name: str,
) -> bool: ...
```
`latest_commit_touching` returns `None` if `relative_path` has no commit history yet. `latest_training_commit` returns `None` if the experiment has no runs, or the most recent one has no `mlflow.source.git.commit` tag. `has_new_data_since_last_training` returns `True` whenever `latest_training_commit` is `None` (nothing trained yet, or never tagged — safest default is "there is something new to use"), otherwise `True` iff the data commit is *not* an ancestor of (and not equal to) the training commit.

- [ ] **Step 1: Write the failing tests**

```python
def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _head(repo):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_latest_commit_touching_returns_none_without_history(tmp_path):
    _git(tmp_path, "init")
    assert retrain_trigger.latest_commit_touching(tmp_path, "data/data_drift") is None


def test_latest_commit_touching_finds_commit_on_path(tmp_path):
    _git(tmp_path, "init")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.txt").write_text("1")
    _git(tmp_path, "add", "data/x.txt")
    _git(tmp_path, "commit", "-m", "add data", "--author=t <t@t>")

    assert retrain_trigger.latest_commit_touching(tmp_path, "data/x.txt") == _head(tmp_path)


class _FakeRun:
    def __init__(self, tags):
        self.data = types.SimpleNamespace(tags=tags)


class _FakeSearchClient:
    def __init__(self, runs):
        self._runs = runs

    def search_runs(self, **kwargs):
        return self._runs


def test_latest_training_commit_reads_tag_from_most_recent_run():
    client = _FakeSearchClient([_FakeRun({"mlflow.source.git.commit": "abc123"})])
    assert retrain_trigger.latest_training_commit(client, "blood_cells-type") == "abc123"


def test_latest_training_commit_none_without_runs():
    client = _FakeSearchClient([])
    assert retrain_trigger.latest_training_commit(client, "blood_cells-type") is None


def test_has_new_data_since_last_training_false_when_already_included(tmp_path):
    _git(tmp_path, "init")
    (tmp_path / "data_drift.txt").write_text("1")
    _git(tmp_path, "add", "data_drift.txt")
    _git(tmp_path, "commit", "-m", "a", "--author=t <t@t>")
    client = _FakeSearchClient([_FakeRun({"mlflow.source.git.commit": _head(tmp_path)})])

    result = retrain_trigger.has_new_data_since_last_training(
        tmp_path, "data_drift.txt", client, "exp"
    )
    assert result is False


def test_has_new_data_since_last_training_true_when_no_prior_run(tmp_path):
    _git(tmp_path, "init")
    (tmp_path / "data_drift.txt").write_text("1")
    _git(tmp_path, "add", "data_drift.txt")
    _git(tmp_path, "commit", "-m", "a", "--author=t <t@t>")
    client = _FakeSearchClient([])

    result = retrain_trigger.has_new_data_since_last_training(
        tmp_path, "data_drift.txt", client, "exp"
    )
    assert result is True


def test_has_new_data_since_last_training_true_when_new_commit_landed(tmp_path):
    _git(tmp_path, "init")
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/experiments/test_retrain_trigger.py -v`
Expected: FAIL

- [ ] **Step 3: Implement the three functions in `src/experiments/retrain_trigger.py`**

`latest_commit_touching`: `subprocess.run(["git", "log", "-1", "--format=%H", "--", relative_path], cwd=repo_dir, capture_output=True, text=True, check=True)`, return `None` if `stdout.strip()` is empty, else that string.

`latest_training_commit`: `client.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"], max_results=1)`; `None` if the list is empty or the first run's `.data.tags` has no `"mlflow.source.git.commit"` key.

`has_new_data_since_last_training`: compute `data_commit = latest_commit_touching(...)` and `training_commit = latest_training_commit(...)`; if `training_commit is None`, return `True`; otherwise run `subprocess.run(["git", "merge-base", "--is-ancestor", data_commit, training_commit], cwd=repo_dir)` **without** `check=True`, and return `result.returncode != 0` (exit code `0` means "already an ancestor," i.e. no new data).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/test_retrain_trigger.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/retrain_trigger.py tests/experiments/test_retrain_trigger.py
git commit -m "feat: add git-commit-based redundant-retraining guard"
```

---

### Task 4: Promotion decision logic with a switchable metric

**Files:**
- Create: `src/experiments/promotion.py`
- Test: `tests/experiments/test_promotion.py` (new)

**Interfaces:**
- Produces:
```python
PROMOTION_METRICS: dict[str, str] = {"f1": "test_f1_macro"}
DEFAULT_PROMOTION_METRIC = "f1"

def decide_promotion(
    champion_scores: dict[str, float],
    challenger_scores: dict[str, float],
    metric: str = DEFAULT_PROMOTION_METRIC,
    regression_margin: float = 0.02,
) -> bool: ...
```
`champion_scores`/`challenger_scores` are always `{"socle": float, "recent": float}` — one already-reduced scalar per eval set. `metric` is accepted and validated (`KeyError`-free lookup against `PROMOTION_METRICS`) but the comparison itself never branches on *which* metric it is — swapping the active metric happens entirely in `PROMOTION_METRICS`/Task 5, never here.

- [ ] **Step 1: Write the failing tests**

```python
def test_decide_promotion_true_when_better_on_recent_and_no_regression():
    assert promotion.decide_promotion(
        {"socle": 0.80, "recent": 0.50}, {"socle": 0.79, "recent": 0.65}
    ) is True


def test_decide_promotion_false_when_worse_on_recent():
    assert promotion.decide_promotion(
        {"socle": 0.80, "recent": 0.60}, {"socle": 0.80, "recent": 0.55}
    ) is False


def test_decide_promotion_false_when_regression_beyond_margin():
    assert promotion.decide_promotion(
        {"socle": 0.80, "recent": 0.50}, {"socle": 0.70, "recent": 0.90}, regression_margin=0.02
    ) is False


def test_decide_promotion_metric_choice_is_a_plain_parameter():
    assert promotion.decide_promotion(
        {"socle": 1.0, "recent": 1.0}, {"socle": 1.0, "recent": 1.0}, metric="criticality_loss"
    ) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/experiments/test_promotion.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `decide_promotion` in `src/experiments/promotion.py`**

```python
def decide_promotion(champion_scores, challenger_scores, metric=DEFAULT_PROMOTION_METRIC, regression_margin=0.02):
    no_regression = challenger_scores["socle"] >= champion_scores["socle"] - regression_margin
    better_on_recent = challenger_scores["recent"] >= champion_scores["recent"]
    return no_regression and better_on_recent
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/test_promotion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/promotion.py tests/experiments/test_promotion.py
git commit -m "feat: add promotion decision logic with switchable metric"
```

---

### Task 5: Evaluation on both held-out sets and alias promotion

**Files:**
- Modify: `src/experiments/promotion.py`
- Test: `tests/experiments/test_promotion.py` (extend)

**Interfaces:**
- Consumes: `decide_promotion`, `PROMOTION_METRICS`, `DEFAULT_PROMOTION_METRIC` (Task 4); `tracking.get_model_from_mlflow` (existing, reused for the champion); `create_dataframe`, `BloodCellDataset`, `DatasetTransform`, `build_metrics` (existing — same imports and call shape as `src/experiments/promote.py`'s own evaluation block, lines ~292-300).
- Produces:
```python
def evaluate_on_eval_sets(
    model: nn.Module, transform, task: TaskSpec, socle_key: str, recent_key: str,
    metric: str = DEFAULT_PROMOTION_METRIC, device: str = "cpu",
) -> dict[str, float]: ...

def get_model_by_run_id(model_name: str, run_id: str, client) -> tuple: ...
# même forme de retour que tracking.get_model_from_mlflow : (transform, model, task, run_id)

def promote_if_better(
    model_name: str, challenger_run_id: str, socle_key: str, recent_key: str,
    metric: str = DEFAULT_PROMOTION_METRIC, client=None,
) -> bool: ...
```

- [ ] **Step 1: Write the failing test for `evaluate_on_eval_sets`**

Build a tiny synthetic eval set under `tmp_path` (2-3 fake images, monkeypatched `DATASET_PATHS` entries for a throwaway dataset key, or reuse `MUNICH_EVAL_KEY`/`DRIFT_EVAL_KEY` pointed at `tmp_path` via monkeypatch) and a trivial `nn.Module` (e.g. a fixed-output linear layer) standing in for a real model, exercising the real `create_dataframe` → `BloodCellDataset` → `DatasetTransform` → `build_metrics` path — same pattern as `src/experiments/promote.py::_train`'s own evaluation block (read that block first; reuse its exact call shape for `build_metrics`/evaluation rather than inventing a new one).

```python
def test_evaluate_on_eval_sets_returns_socle_and_recent_scores(tmp_path, monkeypatch):
    # arrange: images + labels under two monkeypatched DATASET_PATHS entries,
    # a deterministic model, same evaluation call shape as promote.py::_train
    ...
    scores = promotion.evaluate_on_eval_sets(model, transform, task, "munich_eval", "data_drift_eval")
    assert set(scores) == {"socle", "recent"}
    assert 0.0 <= scores["socle"] <= 1.0
    assert 0.0 <= scores["recent"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/test_promotion.py -v`
Expected: FAIL (`AttributeError`)

- [ ] **Step 3: Implement `evaluate_on_eval_sets`**

For each of `socle_key`/`recent_key`: `df = create_dataframe(DataConfig(datasets=[key]), task)`; `ds = DatasetTransform(BloodCellDataset(df, task), transform)`; evaluate with the same `build_metrics(task, for_test=True, device=device)` call shape `promote.py::_train` uses (around its `trainer.evaluate(test_ds, metrics=build_metrics(...))` call); pull `scalars[PROMOTION_METRICS[metric]]` into the returned dict under `"socle"`/`"recent"` respectively.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/experiments/test_promotion.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for `get_model_by_run_id` and `promote_if_better`**

```python
class _FakeRegistryClient(_FakeSearchClient):
    def __init__(self, runs, champion_alias=None):
        super().__init__(runs)
        self._champion_alias = champion_alias
        self.set_alias_calls = []

    def get_model_version_by_alias(self, name, alias):
        if self._champion_alias is None:
            raise mlflow.MlflowException("no champion", error_code=ErrorCode.Name(RESOURCE_DOES_NOT_EXIST))
        return self._champion_alias

    def set_registered_model_alias(self, name, alias, version):
        self.set_alias_calls.append((name, alias, version))


def test_promote_if_better_sets_alias_when_decision_is_true(monkeypatch):
    client = _FakeRegistryClient([], champion_alias=types.SimpleNamespace(version="3", run_id="champ-run"))
    monkeypatch.setattr(promotion, "evaluate_on_eval_sets", lambda *a, **k: (
        {"socle": 0.80, "recent": 0.50} if "champ" in str(a) else {"socle": 0.79, "recent": 0.90}
    ))
    # ... monkeypatch tracking.get_model_from_mlflow / promotion.get_model_by_run_id pour renvoyer
    # des doubles distincts selon qu'on charge le champion ou le challenger (cf. Step 6)

    decision = promotion.promote_if_better(
        "densenet121-type", "challenger-run-id", "munich_eval", "data_drift_eval", client=client
    )

    assert decision is True
    assert client.set_alias_calls == [("densenet121-type", "champion", "3")]


def test_promote_if_better_does_not_set_alias_when_decision_is_false(monkeypatch):
    # même montage, mais evaluate_on_eval_sets renvoie un challenger moins bon
    ...
    decision = promotion.promote_if_better(...)
    assert decision is False
    assert client.set_alias_calls == []
```

(The exact monkeypatching of "which call corresponds to champion vs. challenger" depends on how Step 6 below structures the two `get_model_by_run_id`/`get_model_from_mlflow` calls — write the test against whatever two distinct, clearly-named internal calls Step 6 ends up making, not against positional call order.)

- [ ] **Step 6: Implement `get_model_by_run_id` and `promote_if_better`**

`get_model_by_run_id` mirrors `tracking.get_model_from_mlflow`'s body from the point it has a `run_id` (it already separates "resolve alias → run_id" from "load from run_id" only loosely — factor the "load from run_id" half into a small shared helper if that keeps both functions under control, or duplicate the handful of lines if factoring would require a larger refactor than this task's scope; either is acceptable, prefer sharing if it's a clean extraction).

`promote_if_better`: `client = client or mlflow.MlflowClient()`; load champion via `tracking.get_model_from_mlflow(model_name, client, "champion")` (tolerate "no champion yet" the same way `tracking.get_champion_metric` already does — if there's no champion, the challenger should be promoted unconditionally, no evaluation needed); load challenger via `get_model_by_run_id(model_name, challenger_run_id, client)`; `champion_scores = evaluate_on_eval_sets(champion_model, champion_transform, champion_task, socle_key, recent_key, metric)`, same for challenger; `decision = decide_promotion(champion_scores, challenger_scores, metric)`; if `decision`, `client.set_registered_model_alias(model_name, "champion", <challenger's model version>)` (resolve the version via `client.search_model_versions` filtered on `run_id`, or whatever `get_model_by_run_id` already returns alongside the loaded model); return `decision`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/test_promotion.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/experiments/promotion.py tests/experiments/test_promotion.py
git commit -m "feat: evaluate challenger against champion and promote on decision"
```

---

### Task 6: The Airflow DAG

**Files:**
- Create: `dags/retrain_on_drift.py`

**Interfaces:**
- Consumes: `retrain_trigger.query_drift_share`/`is_drift_sustained`/`has_new_data_since_last_training` (Tasks 2-3); `promotion.promote_if_better` (Tasks 4-5); the `"MLflow run id : "` stdout line (Task 1).

This task has no `pytest` steps: `apache-airflow` is not a dependency of this project's `uv` environment (it runs only inside the `apache/airflow:3.3.2` image via `docker-compose.yaml`), so `dags/retrain_on_drift.py` cannot be imported by this repo's test suite. Correctness here rests on the pure functions already tested in Tasks 2-5; this file only wires them together.

- [ ] **Step 1: Write `dags/retrain_on_drift.py`**

- `@dag(dag_id="retrain_on_drift", schedule="*/10 * * * *", max_active_runs=1, catchup=False)`.
- `@task.branch def check_drift_and_new_data(): ...` — calls `is_drift_sustained(query_drift_share(PROMETHEUS_URL, MODEL_NAME))` and `has_new_data_since_last_training(REPO_DIR, "data/train/data_drift.dvc", mlflow.MlflowClient(), EXPERIMENT_NAME)` (the trainable-pool commit path — see §2's "Portée de la clé munich"-adjacent note in the design doc; eval-only commits under `data/test/data_drift.dvc` correctly don't trigger a retrain since training never reads that data); returns the `"train"` task id if both are true, else `"skip"` (a trivial `EmptyOperator` task).
- `train = DockerOperator(task_id="train", image="blood_cells_training:latest", command=[...configs/densenet_data_drift.yaml...], environment={"GIT_COMMIT": <current HEAD, via `retrain_trigger.latest_commit_touching(REPO_DIR, ".")` or `git rev-parse HEAD`>}, do_xcom_push=True, ...)`.
- `@task def promote(train_output: str): ...` — parses the `"MLflow run id : <id>"` line out of `train_output` (the XCom'd container stdout), calls `promotion.promote_if_better(MODEL_NAME, run_id, "munich_eval", "data_drift_eval")`.
- Wire: `check_drift_and_new_data() >> [train, skip]`; `train >> promote(train.output)`.

- [ ] **Step 2: Manual verification (not automated in this repo)**

`docker compose up -d airflow-webserver airflow-scheduler airflow-worker`; open the Airflow UI; confirm `retrain_on_drift` appears with no DAG import errors; trigger it manually once against a test fixture with known-stale data (expect the `skip` branch) and once with fresh commits ahead of the last training run (expect `train` → `promote` to run to completion). Record the outcome as a note on this task when executing the plan — this step is the task's actual acceptance check, in place of a unit test.

- [ ] **Step 3: Commit**

```bash
git add dags/retrain_on_drift.py
git commit -m "feat: add retrain_on_drift Airflow DAG"
```
