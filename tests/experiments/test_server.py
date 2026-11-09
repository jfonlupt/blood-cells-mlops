"""Gauges Prometheus du service de serving : helper ``_set_drift_gauge`` et cache TTL du champion."""

from types import SimpleNamespace

import bentoml
import mlflow
import pytest
import torch
from PIL import Image

from src.experiments import drift, server


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


def _bare_service():
    # @bentoml.service enveloppe la classe dans un objet Service ; la classe Python
    # réelle (celle qui porte _get_champion) est exposée via .inner sur cette version
    # de bentoml.
    cls = server.BloodCellModelService.inner
    svc = cls.__new__(cls)
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
        # Panne MLflow réaliste (pas un ValueError de registre/config) : c'est précisément
        # ce que la panne-type que ce chemin de secours doit couvrir lève réellement.
        raise mlflow.MlflowException("connection refused")

    flaky.n = 0
    monkeypatch.setattr(drift, "load_champion_reference", flaky)
    clock = iter([0.0, 1000.0, 1050.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(clock))

    assert svc._get_champion("m") == "ref-1"
    with pytest.warns(UserWarning, match="rechargement du champion"):
        # reload a échoué, l'ancienne référence est conservée
        assert svc._get_champion("m") == "ref-1"

    # Le rechargement échoué doit lui aussi rafraîchir _loaded_at : la requête suivante
    # (t=1050, à moins de TTL du dernier essai à t=1000) ne doit PAS retenter un rechargement.
    assert svc._get_champion("m") == "ref-1"
    assert flaky.n == 2


def test_get_champion_ttl_is_independent_per_model(monkeypatch):
    svc = _bare_service()
    calls: dict[str, list] = {"m1": [], "m2": []}
    refs = {"m1": iter(["m1-ref-1", "m1-ref-2"]), "m2": iter(["m2-ref-1", "m2-ref-2"])}

    def fake_load(model, *a, **k):
        calls[model].append(1)
        return next(refs[model])

    monkeypatch.setattr(drift, "load_champion_reference", fake_load)
    # m1 est chargé à t=0, m2 à t=650 (horloge propre). A t=700, le TTL de m1 (démarré à 0)
    # est expiré (700-0=700 > 600) alors que celui de m2 (démarré à 650) ne l'est pas
    # (700-650=50 < 600) : recharger m1 ne doit ni réinitialiser ni autrement affecter
    # l'horloge TTL indépendante de m2.
    clock = iter([0.0, 650.0, 700.0, 700.0])
    monkeypatch.setattr(server.time, "monotonic", lambda: next(clock))

    assert svc._get_champion("m1") == "m1-ref-1"
    assert svc._get_champion("m2") == "m2-ref-1"
    assert svc._get_champion("m1") == "m1-ref-2"  # TTL de m1 expiré -> rechargé
    assert svc._get_champion("m2") == "m2-ref-1"  # TTL de m2 encore frais -> inchangé

    assert len(calls["m1"]) == 2
    assert len(calls["m2"]) == 1


def test_get_champion_raises_on_first_load_failure(monkeypatch):
    svc = _bare_service()
    monkeypatch.setattr(
        drift, "load_champion_reference",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("no model")),
    )

    with pytest.raises(bentoml.exceptions.InvalidArgument):
        svc._get_champion("m")


def test_evaluate_wires_drift_result_into_gauge(monkeypatch, tmp_path):
    """Confirme le câblage evaluate() -> model_drift_detected sur un vrai appel (pas seulement
    _set_drift_gauge en isolation). _get_champion, build_current_dataset et
    compute_drift_report sont mockés ; le reste (predict_images, calcul des métriques de
    classification) s'exécute normalement sur des images/labels minimaux.
    """
    svc = _bare_service()
    class_names = ["a", "b", "c"]
    champion = SimpleNamespace(
        model_name="m",
        task=SimpleNamespace(kind="multiclass", class_names=class_names),
        transform=lambda img: torch.zeros(1),
        model=lambda batch: torch.eye(len(class_names)) * 5.0,
    )
    monkeypatch.setattr(svc, "_get_champion", lambda model: champion)
    monkeypatch.setattr(drift, "build_current_dataset", lambda champion, images: "current-batch")

    fake_metrics = [
        {
            "metric_name": "evidently:metric_v2:DriftedColumnsCount",
            "value": {"count": 1, "share": 0.5},
        }
    ]
    drift_result = drift.DriftMetricsResult(
        descriptors_drifted=True,
        embeddings_drifted=None,
        snapshot=SimpleNamespace(dict=lambda: {"metrics": fake_metrics}),
    )
    monkeypatch.setattr(drift, "compute_drift_report", lambda champion, current: drift_result)

    fake_gauge = _FakeGauge()
    monkeypatch.setattr(server, "model_drift_detected", fake_gauge)

    images = []
    for name in class_names:
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (2, 2)).save(path)
        images.append(path)

    svc.evaluate(model="m", labels=class_names, images=images)

    assert fake_gauge.children[(("model", "m"),)].value == 1.0
