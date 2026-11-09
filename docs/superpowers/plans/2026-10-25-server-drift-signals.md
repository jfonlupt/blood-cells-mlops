# Server Drift Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `blood_cells_server` a single boolean drift signal Airflow can poll, and make its champion cache self-refresh on a TTL instead of loading the champion once forever.

**Architecture:** A pure helper in `drift.py` collapses the already-computed `descriptors_drifted`/`embeddings_drifted` booleans into one flag; `server.py` exposes it as a new Prometheus gauge and reuses the same booleans it already computes today (no new threshold anywhere in `server.py`). `_get_champion`'s cache gains a per-model timestamp and reloads from MLflow when stale, degrading gracefully to the stale entry if the reload itself fails.

**Tech Stack:** Python, `bentoml.metrics` (prometheus client under the hood), `pytest` with hand-rolled fake doubles (project convention — see `tests/experiments/test_drift.py`/`test_tracking.py`).

**Spec:** [docs/superpowers/specs/2026-10-25-data-drift-retraining-design.md](../specs/2026-10-25-data-drift-retraining-design.md)

## Global Constraints

- The new gauge must be named `model_drift_detected`, labeled `["model"]` — fixed by the spec's decision table ("Unification des seuils Evidently/Grafana").
- It must be derived from `compute_drift_report`'s already-computed `descriptors_drifted`/`embeddings_drifted` booleans — never re-derive a numeric threshold inside `server.py`; Evidently/`drift.py` stays the only place a drift threshold is defined.
- The champion cache TTL is env-configurable (`CHAMPION_CACHE_TTL_SECONDS`, default `600`) — the spec leaves the exact value to the implementer, this plan fixes the default and the mechanism.
- On a reload failure with an existing (stale) cache entry, keep serving the stale entry and warn — never turn a transient MLflow outage into a `/predict`/`/evaluate` 404 for a model that was working a moment ago. A failure with *no* prior cache entry still raises exactly as before.

## Review Focus

1. Reload failure (MLflow unreachable) when a stale-but-present cached champion exists must not turn `/predict`/`/evaluate` into a 404 — covered by Task 3's "serves stale on reload failure" test.
2. First-load failure (no cache yet) must still raise `bentoml.exceptions.InvalidArgument` exactly as before — covered by Task 3's "raises on first load failure" test.
3. Two different `model` label values must get independent TTL clocks — reloading one model must not reset or short-circuit another's cache entry. Not covered by a dedicated test below (both tests use a single model name); flagged as a gap the implementer should close by parametrizing one of Task 3's tests over two model names rather than skipping it.
4. `embeddings_drifted is None` (no embeddings section in the report) must count as "not drifted" for that component, not raise — covered by Task 1's `test_is_drift_detected_handles_missing_embeddings_section`.
5. The gauge must reflect exactly what `compute_drift_report` already decided — never a second, independently-coded threshold in `server.py`. Enforced structurally: Task 2's `_set_drift_gauge` only ever calls `drift.is_drift_detected`, it takes no threshold parameter of its own.

---

### Task 1: `is_drift_detected` helper in `drift.py`

**Files:**
- Modify: `src/experiments/drift.py`
- Test: `tests/experiments/test_drift.py`

**Interfaces:**
- Consumes: `DriftMetricsResult` (existing, `src/experiments/drift.py`).
- Produces: `def is_drift_detected(result: DriftMetricsResult) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
def test_is_drift_detected_true_if_descriptors_drifted():
    result = DriftMetricsResult(descriptors_drifted=True, embeddings_drifted=False, snapshot=None)
    assert drift.is_drift_detected(result) is True


def test_is_drift_detected_true_if_embeddings_drifted():
    result = DriftMetricsResult(descriptors_drifted=False, embeddings_drifted=True, snapshot=None)
    assert drift.is_drift_detected(result) is True


def test_is_drift_detected_false_if_neither():
    result = DriftMetricsResult(descriptors_drifted=False, embeddings_drifted=False, snapshot=None)
    assert drift.is_drift_detected(result) is False


def test_is_drift_detected_handles_missing_embeddings_section():
    result = DriftMetricsResult(descriptors_drifted=False, embeddings_drifted=None, snapshot=None)
    assert drift.is_drift_detected(result) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/experiments/test_drift.py -v`
Expected: FAIL (`AttributeError: module 'drift' has no attribute 'is_drift_detected'`)

- [ ] **Step 3: Implement `is_drift_detected` in `src/experiments/drift.py`**

`return bool(result.descriptors_drifted) or bool(result.embeddings_drifted)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/test_drift.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/drift.py tests/experiments/test_drift.py
git commit -m "feat: add is_drift_detected helper"
```

---

### Task 2: `model_drift_detected` gauge in `server.py`

**Files:**
- Modify: `src/experiments/server.py`
- Test: `tests/experiments/test_server.py` (new)

**Interfaces:**
- Consumes: `drift.is_drift_detected` (Task 1).
- Produces: module-level `model_drift_detected = bentoml.metrics.Gauge("model_drift_detected", "1 si une dérive (descripteurs ou embeddings) est détectée, 0 sinon", ["model"])`; `def _set_drift_gauge(gauge, model: str, drift_result: drift.DriftMetricsResult) -> None`. `evaluate()` calls `_set_drift_gauge(model_drift_detected, model, drift_result)` right after its existing `model_descriptor_columns_drifted_*`/`model_embeddings_drift` gauge calls (same `drift_result` already in scope there — `src/experiments/server.py:210-229`).

- [ ] **Step 1: Write the failing tests**

```python
class _FakeGaugeChild:
    def __init__(self):
        self.value = None

    def set(self, v):
        self.value = v


class _FakeGauge:
    def __init__(self):
        self.children = {}

    def labels(self, **kwargs):
        return self.children.setdefault(tuple(sorted(kwargs.items())), _FakeGaugeChild())


def test_set_drift_gauge_sets_one_when_drift_detected():
    gauge = _FakeGauge()
    result = drift.DriftMetricsResult(descriptors_drifted=True, embeddings_drifted=False, snapshot=None)

    server._set_drift_gauge(gauge, "densenet121-type", result)

    assert gauge.children[(("model", "densenet121-type"),)].value == 1.0


def test_set_drift_gauge_sets_zero_when_no_drift():
    gauge = _FakeGauge()
    result = drift.DriftMetricsResult(descriptors_drifted=False, embeddings_drifted=False, snapshot=None)

    server._set_drift_gauge(gauge, "densenet121-type", result)

    assert gauge.children[(("model", "densenet121-type"),)].value == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/experiments/test_server.py -v`
Expected: FAIL (`AttributeError: module 'server' has no attribute '_set_drift_gauge'`)

- [ ] **Step 3: Implement `_set_drift_gauge` and the `model_drift_detected` Gauge in `src/experiments/server.py`**

`_set_drift_gauge(gauge, model, drift_result)` does `gauge.labels(model=model).set(1.0 if drift.is_drift_detected(drift_result) else 0.0)`. Add the call inside `evaluate()`, immediately after the existing embeddings-drift gauge block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/test_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/server.py tests/experiments/test_server.py
git commit -m "feat: expose model_drift_detected prometheus gauge"
```

---

### Task 3: TTL-based champion cache refresh

**Files:**
- Modify: `src/experiments/server.py`
- Test: `tests/experiments/test_server.py` (extend)

**Interfaces:**
- Consumes: `drift.load_champion_reference` (existing, monkeypatched in tests).
- Produces: `BloodCellModelService.__init__` gains `self._loaded_at: dict[str, float] = {}`; module-level `CHAMPION_CACHE_TTL_SECONDS = float(os.environ.get("CHAMPION_CACHE_TTL_SECONDS", "600"))`; `_get_champion(self, model: str)` reloads when `time.monotonic() - self._loaded_at[model] >= CHAMPION_CACHE_TTL_SECONDS` (or when `model` has never been loaded), and on a reload `ValueError` with an existing stale entry, logs a warning via `warnings.warn` and returns the stale entry instead of raising. A `ValueError` with no prior entry still raises `bentoml.exceptions.InvalidArgument` exactly as today.

- [ ] **Step 1: Write the failing tests**

```python
def _bare_service():
    svc = server.BloodCellModelService.__new__(server.BloodCellModelService)
    svc.models, svc._loaded_at, svc.client = {}, {}, None
    return svc


def test_get_champion_reuses_cache_within_ttl(monkeypatch):
    svc = _bare_service()
    calls = []
    monkeypatch.setattr(drift, "load_champion_reference", lambda *a, **k: calls.append(1) or "ref-1")
    clock = iter([0.0, 5.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(clock))

    assert svc._get_champion("m") == "ref-1"
    assert svc._get_champion("m") == "ref-1"
    assert len(calls) == 1


def test_get_champion_reloads_after_ttl(monkeypatch):
    svc = _bare_service()
    refs = iter(["ref-1", "ref-2"])
    monkeypatch.setattr(drift, "load_champion_reference", lambda *a, **k: next(refs))
    clock = iter([0.0, 1000.0])  # 1000 > CHAMPION_CACHE_TTL_SECONDS par défaut (600)
    monkeypatch.setattr(server.time, "monotonic", lambda: next(clock))

    assert svc._get_champion("m") == "ref-1"
    assert svc._get_champion("m") == "ref-2"


def test_get_champion_serves_stale_on_reload_failure(monkeypatch):
    svc = _bare_service()

    def flaky(*a, **k):
        flaky.n += 1
        if flaky.n == 1:
            return "ref-1"
        raise ValueError("mlflow down")

    flaky.n = 0
    monkeypatch.setattr(drift, "load_champion_reference", flaky)
    clock = iter([0.0, 1000.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(clock))

    assert svc._get_champion("m") == "ref-1"
    assert svc._get_champion("m") == "ref-1"  # reload a échoué, l'ancienne référence est conservée


def test_get_champion_raises_on_first_load_failure(monkeypatch):
    svc = _bare_service()
    monkeypatch.setattr(
        drift, "load_champion_reference",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("no model")),
    )

    with pytest.raises(bentoml.exceptions.InvalidArgument):
        svc._get_champion("m")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/experiments/test_server.py -v`
Expected: FAIL

- [ ] **Step 3: Implement the TTL cache in `src/experiments/server.py`**

Add `import time` (top of file, if not already present) and `CHAMPION_CACHE_TTL_SECONDS = float(os.environ.get("CHAMPION_CACHE_TTL_SECONDS", "600"))` near the other module-level constants. In `BloodCellModelService.__init__`, add `self._loaded_at: dict[str, float] = {}`. Rewrite `_get_champion`:

```python
def _get_champion(self, model: str) -> drift.ChampionReference:
    now = time.monotonic()
    fresh = model in self.models and now - self._loaded_at[model] < CHAMPION_CACHE_TTL_SECONDS
    if fresh:
        return self.models[model]
    try:
        self.models[model] = drift.load_champion_reference(model, self.client, "champion")
        self._loaded_at[model] = now
    except ValueError as exc:
        if model in self.models:
            warnings.warn(f"rechargement du champion {model!r} échoué, référence conservée : {exc}", stacklevel=2)
            return self.models[model]
        raise bentoml.exceptions.InvalidArgument(str(exc), error_code=404) from exc
    return self.models[model]
```

(`warnings` needs importing if not already present in `server.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/test_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/experiments/server.py tests/experiments/test_server.py
git commit -m "feat: TTL-based champion cache refresh with graceful degradation"
```
