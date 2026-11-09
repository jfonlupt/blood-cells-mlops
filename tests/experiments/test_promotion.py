"""Décision de promotion champion/challenger avec métrique interchangeable."""

import types

import mlflow
import numpy as np
import pytest
import torch
import yaml
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from PIL import Image
from torch import nn

from src import constants
from src.config import ExperimentConfig
from src.experiments import promotion, tracking
from src.tasks import get_task


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


# ---------------------------------------------------------------------------
# evaluate_on_eval_sets -- pipeline réel create_dataframe -> BloodCellDataset ->
# DatasetTransform -> build_metrics, sur un jeu synthétique minuscule.
# ---------------------------------------------------------------------------


def _write_fake_image(path):
    rng = np.random.default_rng(0)
    Image.fromarray(np.uint8(rng.random((20, 20, 3)) * 255)).save(path)


def _identity_transform(x):
    return x


class _FixedOutputModel(nn.Module):
    """Modèle factice déterministe : ignore le contenu de l'image, renvoie toujours les
    mêmes logits (biaisés vers la classe 0) -- suffisant pour exercer le pipeline
    d'évaluation réel sans dépendre d'un entraînement."""

    def __init__(self, num_classes: int):
        super().__init__()
        logits = torch.zeros(num_classes)
        logits[0] = 5.0
        self.logits = nn.Parameter(logits)

    def forward(self, x):
        return self.logits.unsqueeze(0).expand(x.shape[0], -1)


def test_evaluate_on_eval_sets_returns_socle_and_recent_scores(tmp_path, monkeypatch):
    task = get_task("criticality")

    munich_dir = tmp_path / "munich_eval"
    drift_dir = tmp_path / "drift_eval"
    munich_dir.mkdir()
    drift_dir.mkdir()

    # MUNICH_EVAL_KEY est mappé via MUNICH_TO_FINAL, DRIFT_EVAL_KEY via BARCELONA_TO_FINAL
    # (src/constants.py) -- les codes de fichiers doivent exister dans le mapping du bon dataset.
    #
    # Composition volontairement ASYMÉTRIQUE (et non plus un mélange identique des deux
    # mêmes classes sur les deux jeux) : le socle (munich_eval) ne contient que des
    # cellules critiques (Blast), le jeu récent (data_drift_eval) que des cellules
    # normales (Eosinophil). _FixedOutputModel prédit toujours "normal" -- il se trompe
    # sur tout le socle et a juste sur tout le jeu récent. Si evaluate_on_eval_sets
    # intervertissait par erreur quel dataframe alimente "socle" vs "recent", les deux
    # scores seraient inversés plutôt qu'identiques : avec l'ancienne composition
    # symétrique, une telle inversion ne changeait rien et passait silencieusement.
    for i in range(3):
        _write_fake_image(munich_dir / f"MYO_{i}.jpg")  # Blast -> critique (MUNICH_TO_FINAL)
    for i in range(3):
        _write_fake_image(drift_dir / f"EO_{i}.jpg")  # Eosinophil -> normal (BARCELONA_TO_FINAL)

    monkeypatch.setitem(constants.DATASET_PATHS, constants.MUNICH_EVAL_KEY, munich_dir)
    monkeypatch.setitem(constants.DATASET_PATHS, constants.DRIFT_EVAL_KEY, drift_dir)

    model = _FixedOutputModel(task.num_classes)

    scores = promotion.evaluate_on_eval_sets(
        model, _identity_transform, task, constants.MUNICH_EVAL_KEY, constants.DRIFT_EVAL_KEY
    )

    assert set(scores) == {"socle", "recent"}
    assert 0.0 <= scores["socle"] <= 1.0
    assert 0.0 <= scores["recent"] <= 1.0
    # Le modèle prédit toujours "normal" : 0/3 juste sur le socle (que des critiques),
    # 3/3 justes sur le jeu récent (que des normaux) -- recent doit nettement dominer.
    assert scores["socle"] == pytest.approx(0.0, abs=1e-6)
    assert scores["recent"] > scores["socle"]


# ---------------------------------------------------------------------------
# get_model_by_run_id -- même garde-fous que tracking.get_model_from_mlflow,
# résolution par run_id (search_model_versions) plutôt que par alias.
# ---------------------------------------------------------------------------


class _FakeModelVersion:
    def __init__(self, name="blood_cells-type", run_id="run123", version="7", source="models:/foo/7"):
        self.name = name
        self.run_id = run_id
        self.version = version
        self.source = source


class _FakeSearchVersionsClient:
    """Double minimal qui respecte le filtre ``name``/``run_id``, comme le vrai serveur.

    Ignorer ``filter_string`` (comme une première version de ce double le faisait)
    masquerait une régression sur le scoping par ``name`` -- cf. revue.
    """

    def __init__(self, versions):
        self._versions = versions

    def search_model_versions(self, filter_string):
        return [
            mv
            for mv in self._versions
            if f"name='{mv.name}'" in filter_string and f"run_id='{mv.run_id}'" in filter_string
        ]


class _FakeModelWithFeatures(torch.nn.Module):
    """Vrai ``nn.Module`` exposant ``.features()`` -- voir le garde-fou correspondant
    dans ``_load_model_version`` (mêmes raisons que dans ``tracking.py``)."""

    def features(self, x):
        return x


def test_get_model_by_run_id_returns_model_task_transform_and_run_id(monkeypatch):
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
        lambda uri: {
            "dataset_mean": [0.5, 0.5, 0.5],
            "dataset_std": [0.25, 0.25, 0.25],
            "reinhard": None,
        },
    )
    client = _FakeSearchVersionsClient(
        [_FakeModelVersion(name="blood_cells-type", run_id="run123", version="7")]
    )

    transform, model, task, run_id = promotion.get_model_by_run_id(
        "blood_cells-type", "run123", client
    )

    assert run_id == "run123"
    assert model is fake_model
    assert task.name == "type"
    out = transform(torch.randint(0, 255, (3, 30, 30), dtype=torch.uint8))
    assert out.shape == (3, 224, 224)


def test_get_model_by_run_id_raises_value_error_when_run_id_not_registered():
    client = _FakeSearchVersionsClient([])

    with pytest.raises(ValueError, match="run_id"):
        promotion.get_model_by_run_id("blood_cells-type", "missing-run", client)


def test_get_model_by_run_id_uses_model_version_without_searching_when_given(monkeypatch):
    """``model_version`` déjà résolu (cf. ``promote_if_better``) : pas de recherche redondante."""
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
        lambda uri: {
            "dataset_mean": [0.5, 0.5, 0.5],
            "dataset_std": [0.25, 0.25, 0.25],
            "reinhard": None,
        },
    )

    def _fail_search(filter_string):
        raise AssertionError("ne doit pas rechercher : model_version déjà fourni")

    client = types.SimpleNamespace(search_model_versions=_fail_search)
    model_version = _FakeModelVersion(name="blood_cells-type", run_id="run123", version="7")

    _, model, _, run_id = promotion.get_model_by_run_id(
        "blood_cells-type", "run123", client, model_version=model_version
    )

    assert run_id == "run123"
    assert model is fake_model


def test_find_registered_model_version_filters_by_name_and_picks_highest_version():
    """Finding 1 (revue) : scoping par name ET run_id, et sélection de la version la plus
    récente en cas de doublon -- pas une correspondance arbitraire dépendant de l'ordre."""
    registry = [
        _FakeModelVersion(name="densenet121-type", run_id="run-abc", version="5"),
        _FakeModelVersion(name="densenet121-type", run_id="run-abc", version="7"),
        _FakeModelVersion(name="other-model", run_id="run-abc", version="99"),
    ]
    client = _FakeSearchVersionsClient(registry)

    mv = promotion._find_registered_model_version(client, "densenet121-type", "run-abc")

    assert mv.version == "7"


def test_find_registered_model_version_returns_none_when_absent():
    client = _FakeSearchVersionsClient([])

    assert promotion._find_registered_model_version(client, "densenet121-type", "run-abc") is None


# ---------------------------------------------------------------------------
# promote_if_better -- double mlflow factice, evaluate_on_eval_sets monkeypatché.
# ---------------------------------------------------------------------------


class _FakeRegistryClient:
    """Double MLflow pour ``promote_if_better``.

    ``search_model_versions`` respecte réellement ``filter_string`` (filtré sur
    ``name`` ET ``run_id``, comme le vrai serveur) plutôt que de l'ignorer --
    un double qui l'ignorerait ne détecterait jamais une régression sur le
    scoping (Finding 1 de la revue). ``search_calls`` permet de vérifier que
    la version n'est résolue qu'une seule fois par appel (pas de double
    recherche, Finding 1 également).
    """

    def __init__(self, champion_version=None, registry=()):
        self._champion_version = champion_version
        self._registry = list(registry)
        self.set_alias_calls = []
        self.search_calls = []

    def get_model_version_by_alias(self, name, alias):
        if self._champion_version is None:
            raise mlflow.MlflowException("no champion", error_code=RESOURCE_DOES_NOT_EXIST)
        return types.SimpleNamespace(version=self._champion_version, run_id="champ-run")

    def get_run(self, run_id):
        return types.SimpleNamespace(data=types.SimpleNamespace(metrics={}))

    def search_model_versions(self, filter_string):
        self.search_calls.append(filter_string)
        return [
            mv
            for mv in self._registry
            if f"name='{mv.name}'" in filter_string and f"run_id='{mv.run_id}'" in filter_string
        ]

    def set_registered_model_alias(self, name, alias, version):
        self.set_alias_calls.append((name, alias, version))


_CHALLENGER_RUN_ID = "challenger-run-id"
_CHAMPION_MODEL = object()
_CHALLENGER_MODEL = object()
_CHAMPION_TASK = types.SimpleNamespace(name="type")
_CHALLENGER_TASK = types.SimpleNamespace(name="type")


def _fake_get_model_from_mlflow(model_name, client, alias):
    assert alias == "champion"
    return "champion-transform", _CHAMPION_MODEL, _CHAMPION_TASK, "champ-run"


def _fake_get_model_by_run_id(model_name, run_id, client, model_version=None):
    assert run_id == _CHALLENGER_RUN_ID
    return "challenger-transform", _CHALLENGER_MODEL, _CHALLENGER_TASK, run_id


def _make_fake_evaluate(champion_scores, challenger_scores):
    def fake_evaluate(
        model,
        transform,
        task,
        socle_key,
        recent_key,
        metric=promotion.DEFAULT_PROMOTION_METRIC,
        device="cpu",
    ):
        if model is _CHAMPION_MODEL:
            return dict(champion_scores)
        if model is _CHALLENGER_MODEL:
            return dict(challenger_scores)
        raise AssertionError(f"modèle inattendu passé à evaluate_on_eval_sets: {model!r}")

    return fake_evaluate


def _registered_challenger(version="7"):
    return _FakeModelVersion(
        name="densenet121-type", run_id=_CHALLENGER_RUN_ID, version=version
    )


def test_promote_if_better_sets_alias_when_decision_is_true(monkeypatch):
    client = _FakeRegistryClient(champion_version="3", registry=[_registered_challenger("7")])
    monkeypatch.setattr(tracking, "get_model_from_mlflow", _fake_get_model_from_mlflow)
    monkeypatch.setattr(promotion, "get_model_by_run_id", _fake_get_model_by_run_id)
    monkeypatch.setattr(
        promotion,
        "evaluate_on_eval_sets",
        _make_fake_evaluate({"socle": 0.80, "recent": 0.50}, {"socle": 0.79, "recent": 0.90}),
    )

    decision = promotion.promote_if_better(
        "densenet121-type", _CHALLENGER_RUN_ID, "munich_eval", "data_drift_eval", client=client
    )

    assert decision is True
    assert client.set_alias_calls == [("densenet121-type", "champion", "7")]
    # Une seule recherche de version pour tout l'appel (pas une pour le chargement et
    # une autre, potentiellement différente, pour l'alias -- Finding 1 de la revue).
    assert len(client.search_calls) == 1


def test_promote_if_better_does_not_set_alias_when_decision_is_false(monkeypatch):
    client = _FakeRegistryClient(champion_version="3", registry=[_registered_challenger("7")])
    monkeypatch.setattr(tracking, "get_model_from_mlflow", _fake_get_model_from_mlflow)
    monkeypatch.setattr(promotion, "get_model_by_run_id", _fake_get_model_by_run_id)
    monkeypatch.setattr(
        promotion,
        "evaluate_on_eval_sets",
        _make_fake_evaluate({"socle": 0.80, "recent": 0.60}, {"socle": 0.80, "recent": 0.55}),
    )

    decision = promotion.promote_if_better(
        "densenet121-type", _CHALLENGER_RUN_ID, "munich_eval", "data_drift_eval", client=client
    )

    assert decision is False
    assert client.set_alias_calls == []


def test_promote_if_better_promotes_unconditionally_when_no_champion_yet(monkeypatch):
    client = _FakeRegistryClient(champion_version=None, registry=[_registered_challenger("1")])

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("ne doit pas être appelé sans champion à comparer")

    monkeypatch.setattr(tracking, "get_model_from_mlflow", _fail_if_called)
    monkeypatch.setattr(promotion, "get_model_by_run_id", _fail_if_called)
    monkeypatch.setattr(promotion, "evaluate_on_eval_sets", _fail_if_called)

    decision = promotion.promote_if_better(
        "densenet121-type", _CHALLENGER_RUN_ID, "munich_eval", "data_drift_eval", client=client
    )

    assert decision is True
    assert client.set_alias_calls == [("densenet121-type", "champion", "1")]


def test_promote_if_better_registers_challenger_when_not_already_registered(monkeypatch):
    """Finding 2 (revue) : tracking.register_model ne crée une version que si le run
    franchit son propre gate F1 (standard split) -- indépendant du socle/jeu récent
    évalués ici. Un challenger de réentraînement sur drift peut donc n'avoir jamais
    été enregistré malgré de bonnes performances sur le jeu récent ; promote_if_better
    doit l'enregistrer lui-même plutôt que planter."""
    client = _FakeRegistryClient(champion_version="3", registry=[])  # challenger absent du registre
    monkeypatch.setattr(tracking, "get_model_from_mlflow", _fake_get_model_from_mlflow)
    monkeypatch.setattr(promotion, "get_model_by_run_id", _fake_get_model_by_run_id)
    monkeypatch.setattr(
        promotion,
        "evaluate_on_eval_sets",
        _make_fake_evaluate({"socle": 0.80, "recent": 0.50}, {"socle": 0.79, "recent": 0.90}),
    )

    register_calls = []
    registered = types.SimpleNamespace(
        name="densenet121-type", version="9", run_id=_CHALLENGER_RUN_ID
    )

    def _fake_register_model(model_uri, name):
        register_calls.append((model_uri, name))
        return registered

    monkeypatch.setattr(promotion.mlflow, "register_model", _fake_register_model)

    decision = promotion.promote_if_better(
        "densenet121-type", _CHALLENGER_RUN_ID, "munich_eval", "data_drift_eval", client=client
    )

    assert decision is True
    assert register_calls == [(f"runs:/{_CHALLENGER_RUN_ID}/model", "densenet121-type")]
    assert client.set_alias_calls == [("densenet121-type", "champion", "9")]


def test_promote_if_better_raises_when_champion_and_challenger_tasks_differ(monkeypatch):
    """Bundle 6 (revue finale) : comparer des scores de tâches différentes (ex. "type" vs
    "criticality") serait silencieusement incorrect -- doit échouer bruyamment plutôt que
    de laisser decide_promotion comparer des métriques incomparables."""
    client = _FakeRegistryClient(champion_version="3", registry=[_registered_challenger("7")])

    def _fake_get_model_from_mlflow_other_task(model_name, client, alias):
        assert alias == "champion"
        return (
            "champion-transform",
            _CHAMPION_MODEL,
            types.SimpleNamespace(name="criticality"),
            "champ-run",
        )

    monkeypatch.setattr(tracking, "get_model_from_mlflow", _fake_get_model_from_mlflow_other_task)
    monkeypatch.setattr(promotion, "get_model_by_run_id", _fake_get_model_by_run_id)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("ne doit pas évaluer des modèles de tâches incomparables")

    monkeypatch.setattr(promotion, "evaluate_on_eval_sets", _fail_if_called)

    with pytest.raises(ValueError, match="comparables"):
        promotion.promote_if_better(
            "densenet121-type", _CHALLENGER_RUN_ID, "munich_eval", "data_drift_eval", client=client
        )

    assert client.set_alias_calls == []


def test_promote_if_better_passes_through_device_and_regression_margin(monkeypatch):
    client = _FakeRegistryClient(champion_version="3", registry=[_registered_challenger("7")])
    monkeypatch.setattr(tracking, "get_model_from_mlflow", _fake_get_model_from_mlflow)
    monkeypatch.setattr(promotion, "get_model_by_run_id", _fake_get_model_by_run_id)

    evaluate_devices = []

    def fake_evaluate(
        model,
        transform,
        task,
        socle_key,
        recent_key,
        metric=promotion.DEFAULT_PROMOTION_METRIC,
        device="cpu",
    ):
        evaluate_devices.append(device)
        if model is _CHAMPION_MODEL:
            return {"socle": 0.80, "recent": 0.50}
        return {"socle": 0.70, "recent": 0.65}  # régresse sur le socle au-delà d'une petite marge

    monkeypatch.setattr(promotion, "evaluate_on_eval_sets", fake_evaluate)

    decide_margins = []
    real_decide = promotion.decide_promotion

    def spy_decide(champion_scores, challenger_scores, metric=promotion.DEFAULT_PROMOTION_METRIC, regression_margin=0.02):
        decide_margins.append(regression_margin)
        return real_decide(champion_scores, challenger_scores, metric, regression_margin)

    monkeypatch.setattr(promotion, "decide_promotion", spy_decide)

    decision = promotion.promote_if_better(
        "densenet121-type",
        _CHALLENGER_RUN_ID,
        "munich_eval",
        "data_drift_eval",
        client=client,
        device="cuda",
        regression_margin=0.5,
    )

    assert evaluate_devices == ["cuda", "cuda"]
    assert decide_margins == [0.5]
    # Avec regression_margin=0.5, une régression de 0.10 sur le socle est tolérée.
    assert decision is True
