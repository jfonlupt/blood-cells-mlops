"""Rapport de data drift : chargement de la référence, construction du rapport."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from src.experiments import drift
from src.experiments.drift import DriftMetricsResult, DriftReportResult, run_drift_report


class _FakeFileInfo:
    """Double minimal de ``mlflow.entities.FileInfo`` : seul ``.path`` est lu par
    ``drift._artifact_exists`` (vérifié empiriquement contre mlflow==3.17.1, backend
    fichier local : ``client.list_artifacts(run_id, path=...)`` renvoie une liste de
    ``FileInfo``, vide -- sans lever -- aussi bien quand le sous-dossier n'existe pas
    du tout que quand seul le fichier demandé y manque)."""

    def __init__(self, path):
        self.path = path


class _FakeListArtifactsClient:
    """Double de ``mlflow.MlflowClient`` ne fournissant que ``list_artifacts``."""

    def __init__(self, paths):
        self._paths = paths

    def list_artifacts(self, run_id, path=None):
        return [_FakeFileInfo(p) for p in self._paths]


def _patch_list_artifacts(monkeypatch, *, present: bool, filename: str):
    """Simule ``drift_reference/<filename>.parquet`` présent ou absent côté MLflow."""
    paths = [f"drift_reference/{filename}.parquet"] if present else []
    monkeypatch.setattr(drift.mlflow, "MlflowClient", lambda: _FakeListArtifactsClient(paths))


def test_load_required_reference_reads_parquet(tmp_path, monkeypatch):
    local = tmp_path / "descriptors.parquet"
    pd.DataFrame({"path": ["a.png"], "brightness": [0.5]}).to_parquet(local)
    _patch_list_artifacts(monkeypatch, present=True, filename="descriptors")
    monkeypatch.setattr(drift.mlflow.artifacts, "download_artifacts", lambda artifact_uri: str(local))
    df = drift._load_required_reference("run123", "descriptors")
    assert df["brightness"].iloc[0] == 0.5


def test_load_required_reference_raises_clear_error_when_missing(monkeypatch):
    _patch_list_artifacts(monkeypatch, present=False, filename="descriptors")

    def _boom(artifact_uri):
        raise AssertionError("download_artifacts ne doit pas être appelé si l'artefact est absent")

    monkeypatch.setattr(drift.mlflow.artifacts, "download_artifacts", _boom)
    with pytest.raises(FileNotFoundError, match="run123"):
        drift._load_required_reference("run123", "descriptors")


def test_load_optional_reference_returns_none_when_missing(monkeypatch):
    _patch_list_artifacts(monkeypatch, present=False, filename="embeddings")

    def _boom(artifact_uri):
        raise AssertionError("download_artifacts ne doit pas être appelé si l'artefact est absent")

    monkeypatch.setattr(drift.mlflow.artifacts, "download_artifacts", _boom)
    assert drift._load_optional_reference("run123", "embeddings") is None


def test_load_optional_reference_reads_parquet_when_present(tmp_path, monkeypatch):
    local = tmp_path / "embeddings.parquet"
    pd.DataFrame({"path": ["a.png"], "emb_0": [1.0]}).to_parquet(local)
    _patch_list_artifacts(monkeypatch, present=True, filename="embeddings")
    monkeypatch.setattr(drift.mlflow.artifacts, "download_artifacts", lambda artifact_uri: str(local))
    df = drift._load_optional_reference("run123", "embeddings")
    assert df["emb_0"].iloc[0] == 1.0


def _write_images(tmp_path, n, color, seed=0, noise_std=20):
    """Images bruitées (pas un flat uniforme) : chaque canal garde une vraie variance
    intra-groupe, sinon son histogramme à 8 bins est toujours "one-hot" et au plus
    2/8 bins par canal peuvent différer entre deux couleurs, quelle que soit leur
    distance — plafonnant la part de colonnes en drift bien sous le seuil majoritaire
    de ``DatasetDriftMetric`` (0.5), indépendamment du vrai changement de distribution.

    ``noise_std`` par défaut (20) reproduit le comportement historique de ce helper
    (tous les tests existants en dépendent) ; un ``noise_std`` plus élevé (ex. 40)
    est utilisé par le test de régression Fix 1 pour reproduire fidèlement les
    conditions mesurées par le reviewer (faux positif Wasserstein systématique).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        p = tmp_path / f"img_{i}.png"
        noise = rng.normal(0, noise_std, size=(30, 30, 3))
        arr = np.clip(np.asarray(color, dtype=np.float64) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    return paths


# EmbeddingsDriftMetric (méthode "model" par défaut) fait une PCA à 2 composantes
# en interne -- une seule colonne d'embedding (n_features=1) fait planter sklearn
# (n_components=2 > n_features=1). 4 dimensions donne de la marge, comme un vrai
# embedding (jamais mono-dimensionnel en pratique).
_EMBEDDING_DIM = 4


class _FakeModelWithFeatures(nn.Module):
    """Modèle factice exposant ``.features()`` — ``object()`` n'a ni ``.features()``
    ni ``.training``/``.eval()``/``.train()`` qu'utilise ``build_drift_reference``
    (Tâche 8), donc il faut un vrai ``nn.Module``. Même pattern que
    ``_ModelWithFeatures`` dans ``tests/experiments/test_drift_reference.py``.

    Un bruit indépendant par ligne/dimension (plutôt qu'une constante, ou une
    valeur dérivée de l'image et répétée sur toutes les dimensions) : la
    matrice de covariance serait alors singulière, ce qui fait planter la KDE
    gaussienne interne d'``EmbeddingsDriftMetric``.
    """

    def __init__(self, seed=1):
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def features(self, x):
        values = self._rng.normal(0.0, 0.01, size=(x.shape[0], _EMBEDDING_DIM))
        return torch.tensor(values, dtype=torch.float32)

    def forward(self, x):
        return self.features(x)


def _patch_champion(monkeypatch, tmp_path, *, with_embeddings):
    import src.experiments.drift as drift_module

    # mlflow.MlflowClient() est construit (et résout son tracking store) avant même
    # que get_model_from_mlflow (mocké juste après) ne soit appelé — sans
    # MLFLOW_TRACKING_URI, mlflow retombe sur un défaut sqlite que mlflow-skinny ne
    # sait pas gérer. Même convention que tests/experiments/test_tracking.py.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    # object() n'a pas .features() : build_drift_reference ne produit alors jamais
    # d'embeddings, quel que soit with_embeddings -- il faut un modèle réel pour
    # couvrir la branche has_embeddings=True de run_drift_report.
    fake_model = _FakeModelWithFeatures() if with_embeddings else object()

    def _fake_get_model(model_name, client, alias="champion"):
        return (lambda img: img.float() / 255.0), fake_model, None, "run123"

    monkeypatch.setattr(drift_module, "get_model_from_mlflow", _fake_get_model)

    ref_paths = _write_images(tmp_path / "ref", 20, color=(200, 200, 200))  # clair
    descriptors_ref = compute_descriptors_for_test(ref_paths)

    def _fake_required(run_id, filename):
        assert filename == "descriptors"
        return descriptors_ref

    def _fake_optional(run_id, filename):
        if not with_embeddings:
            return None
        emb = descriptors_ref[["path"]].copy()
        # Même raison que _FakeModelWithFeatures.features : bruit indépendant par
        # ligne/dimension pour éviter une matrice de covariance singulière.
        rng = np.random.default_rng(0)
        values = rng.normal(0.1, 0.01, size=(len(emb), _EMBEDDING_DIM))
        for i in range(_EMBEDDING_DIM):
            emb[f"emb_{i}"] = values[:, i]
        return emb

    monkeypatch.setattr(drift_module, "_load_required_reference", _fake_required)
    monkeypatch.setattr(drift_module, "_load_optional_reference", _fake_optional)


def compute_descriptors_for_test(paths):
    from src.features.drift_descriptors import compute_descriptors

    return compute_descriptors(paths)


def test_run_drift_report_detects_drift_on_descriptors(tmp_path, monkeypatch):
    _patch_champion(monkeypatch, tmp_path, with_embeddings=False)
    current_paths = _write_images(
        tmp_path / "current", 20, color=(5, 5, 5), seed=1
    )  # sombre : drift net
    result = run_drift_report(
        "blood_cells-type", current_paths, output=tmp_path / "out"
    )
    assert isinstance(result, DriftReportResult)
    assert result.descriptors_drifted is True
    assert result.embeddings_drifted is None
    assert result.report_path.exists()


def test_run_drift_report_no_drift_when_distributions_match(tmp_path, monkeypatch):
    _patch_champion(monkeypatch, tmp_path, with_embeddings=False)
    # même couleur, tirage de bruit différent (seed=1) : même distribution, échantillon distinct
    current_paths = _write_images(tmp_path / "current", 20, color=(200, 200, 200), seed=1)
    result = run_drift_report(
        "blood_cells-type", current_paths, output=tmp_path / "out"
    )
    assert result.descriptors_drifted is False


def test_run_drift_report_skips_embeddings_section_when_reference_absent(tmp_path, monkeypatch):
    _patch_champion(monkeypatch, tmp_path, with_embeddings=False)
    current_paths = _write_images(tmp_path / "current", 20, color=(200, 200, 200), seed=1)
    result = run_drift_report("blood_cells-type", current_paths, output=tmp_path / "out")
    assert result.embeddings_drifted is None


def test_run_drift_report_includes_embeddings_section_when_reference_present(tmp_path, monkeypatch):
    """Couvre la branche has_embeddings=True : merge on="path", column_mapping.embeddings,
    extraction du résultat EmbeddingsDriftMetric — jamais exercée si le modèle factice
    n'a pas de .features() (cf. _FakeModelWithFeatures).
    """
    _patch_champion(monkeypatch, tmp_path, with_embeddings=True)
    current_paths = _write_images(tmp_path / "current", 20, color=(200, 200, 200), seed=1)
    result = run_drift_report("blood_cells-type", current_paths, output=tmp_path / "out")
    assert result.embeddings_drifted is not None
    assert isinstance(result.embeddings_drifted, bool)


def test_load_champion_reference_returns_populated_champion_reference(tmp_path, monkeypatch):
    """load_champion_reference est le point d'entrée partagé CLI/service (server.py le met
    en cache par nom de modèle) : vérifie que tous les champs utiles sont bien peuplés.
    """
    _patch_champion(monkeypatch, tmp_path, with_embeddings=True)

    champion = drift.load_champion_reference("blood_cells-type", object(), "champion")

    assert isinstance(champion, drift.ChampionReference)
    assert champion.run_id == "run123"
    assert champion.has_embeddings is True
    assert champion.embeddings_ref is not None
    assert "path" not in champion.numerical_features


def test_build_current_dataset_and_compute_drift_report_match_run_drift_report(
    tmp_path, monkeypatch
):
    """server.py appelle build_current_dataset()/compute_drift_report() séparément (il
    garde le ChampionReference en cache entre les requêtes) plutôt que run_drift_report()
    (chemin CLI, qui recharge tout à chaque appel) -- les deux doivent donner le même
    résultat pour les mêmes entrées, puisque run_drift_report n'est qu'une fine couche
    de sauvegarde fichier par-dessus les mêmes fonctions.
    """
    _patch_champion(monkeypatch, tmp_path, with_embeddings=False)
    current_paths = _write_images(tmp_path / "current", 20, color=(5, 5, 5), seed=1)

    champion = drift.load_champion_reference("blood_cells-type", object(), "champion")
    current = drift.build_current_dataset(champion, current_paths)
    result = drift.compute_drift_report(champion, current)

    assert isinstance(result, drift.DriftMetricsResult)
    assert result.descriptors_drifted is True
    assert result.embeddings_drifted is None


def test_compute_drift_report_includes_extra_metrics(tmp_path, monkeypatch):
    """Point d'extension pour une future métrique de qualité de classification dans
    server.py /evaluate (cf. commentaire dans evaluate()) : toute métrique evidently
    supplémentaire passée via extra_metrics doit apparaître dans le snapshot final,
    sans que compute_drift_report() ait besoin d'être modifié pour ça.
    """
    from evidently.metrics import ColumnCount

    _patch_champion(monkeypatch, tmp_path, with_embeddings=False)
    current_paths = _write_images(tmp_path / "current", 20, color=(200, 200, 200), seed=1)

    champion = drift.load_champion_reference("blood_cells-type", object(), "champion")
    current = drift.build_current_dataset(champion, current_paths)
    result = drift.compute_drift_report(champion, current, extra_metrics=[ColumnCount()])

    summary = result.snapshot.dict()
    assert any(m["config"]["type"] == "evidently:metric_v2:ColumnCount" for m in summary["metrics"])


def test_run_drift_report_degrades_gracefully_with_too_few_current_images_for_embeddings(
    tmp_path, monkeypatch
):
    """Régression prod : EmbeddingsDrift calcule son score puis plante sur sa SEULE
    visualisation (projection t-SNE à 2D + gaussian_kde sur les points courants) --
    mathématiquement dégénéré avec moins de 3 images courantes (2 points ne peuvent
    jamais définir une covariance 2D de rang plein). Le rapport doit se dégrader
    gracieusement (comme "pas d'embeddings du tout") plutôt que de tout faire échouer,
    y compris la section descripteurs qui n'est elle pas concernée par ce problème.
    """
    _patch_champion(monkeypatch, tmp_path, with_embeddings=True)
    current_paths = _write_images(tmp_path / "current", 2, color=(200, 200, 200), seed=1)

    with pytest.warns(UserWarning, match="embeddings absente"):
        result = run_drift_report("blood_cells-type", current_paths, output=tmp_path / "out")

    assert result.embeddings_drifted is None
    assert isinstance(result.descriptors_drifted, bool)
    assert result.report_path.exists()


def test_run_drift_report_raises_on_empty_image_list(tmp_path, monkeypatch):
    _patch_champion(monkeypatch, tmp_path, with_embeddings=False)
    with pytest.raises(ValueError, match="aucune image"):
        run_drift_report("blood_cells-type", [], output=tmp_path / "out")


def test_run_drift_report_propagates_missing_reference_error(tmp_path, monkeypatch):
    """La référence manquante doit remonter jusqu'à l'appelant de run_drift_report, pas être avalée."""
    import src.experiments.drift as drift_module

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    def _fake_get_model(model_name, client, alias="champion"):
        return (lambda img: img.float() / 255.0), object(), None, "run123"

    monkeypatch.setattr(drift_module, "get_model_from_mlflow", _fake_get_model)

    # Référence absente, simulée au niveau de la vérification d'existence (spec §3.2) :
    # la liste des artefacts du run ne contient pas drift_reference/descriptors.parquet.
    _patch_list_artifacts(monkeypatch, present=False, filename="descriptors")

    def _boom(artifact_uri):
        raise OSError("download_artifacts ne doit pas être appelé si l'artefact est absent")

    # _load_required_reference n'est PAS mocké ici : on vérifie la composition réelle
    # get_model_from_mlflow -> _load_required_reference -> _artifact_exists -> (absent,
    # FileNotFoundError avant même d'atteindre mlflow.artifacts.download_artifacts).
    monkeypatch.setattr(drift_module.mlflow.artifacts, "download_artifacts", _boom)

    current_paths = _write_images(tmp_path / "current", 5, color=(10, 10, 10))
    with pytest.raises(FileNotFoundError, match="run123"):
        run_drift_report("blood_cells-type", current_paths, output=tmp_path / "out")


def test_run_drift_report_propagates_real_infra_failure_from_download(tmp_path, monkeypatch):
    """Si l'artefact existe (list_artifacts le confirme) mais que le téléchargement
    échoue (réseau/auth/S3), l'erreur doit propager telle quelle -- pas être avalée ni
    réinterprétée comme ``FileNotFoundError`` (spec §3.2/§5 : le try/except générique
    autour de download_artifacts a été retiré précisément pour ça)."""
    import src.experiments.drift as drift_module

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    def _fake_get_model(model_name, client, alias="champion"):
        return (lambda img: img.float() / 255.0), object(), None, "run123"

    monkeypatch.setattr(drift_module, "get_model_from_mlflow", _fake_get_model)
    _patch_list_artifacts(monkeypatch, present=True, filename="descriptors")

    def _infra_failure(artifact_uri):
        raise ConnectionError("S3 injoignable")

    monkeypatch.setattr(drift_module.mlflow.artifacts, "download_artifacts", _infra_failure)

    current_paths = _write_images(tmp_path / "current", 5, color=(10, 10, 10))
    with pytest.raises(ConnectionError, match="S3 injoignable"):
        run_drift_report("blood_cells-type", current_paths, output=tmp_path / "out")


def test_run_drift_report_no_false_positive_with_large_reference_same_distribution(
    tmp_path, monkeypatch
):
    """Régression Fix 1 (revue finale) : avec les défauts Evidently, ``DataDriftPreset()``
    bascule le test numérique par colonne de Kolmogorov-Smirnov vers une distance de
    Wasserstein normée à seuil fixe 0.1 dès que la référence dépasse 1000 lignes -- ce
    test est en faux positif quasi systématique sur un petit lot courant, même sans
    drift réel (mesuré par le reviewer : 100% de faux positifs à 20 images courantes).
    Notre référence réelle (200 images/classe) dépasse 1000 lignes dès 5-6 classes.

    La référence est construite avec exactement le même générateur d'images que le lot
    courant (``_write_images``, même couleur, même ``noise_std``), juste avec 1200
    images (> 1000, pas de sous-échantillonnage synthétique ad hoc qui risquerait de
    déformer la forme réelle de la distribution -- ex. colonnes d'histogramme bornées
    dans [0, 1]) et un tirage de bruit différent (seed différent) -- donc *pas* de drift
    réel, juste un échantillon distinct de la même distribution. ``noise_std=40``
    (au lieu du défaut 20 du helper) est nécessaire : vérifié empiriquement (script ad
    hoc, 24 combinaisons de seeds) que c'est à partir de cette amplitude de bruit que
    le faux positif Wasserstein devient robuste et systématique sur ce jeu de
    descripteurs précis (``dataset_drift=True`` avec les défauts Evidently, dans 100%
    des 24 combinaisons testées, contre ``False`` avec ``num_stattest="ks"`` dans les
    24) -- à bruit plus faible (20), le partage de colonnes en faux-drift restait sous
    le seuil majoritaire de ``DatasetDriftMetric`` (0.5) à cause de plusieurs colonnes
    triviales (``width``/``height``/``aspect_ratio``, constantes ici) qui diluent la
    part de colonnes réellement affectées. Le lot courant, lui, passe par le vrai
    ``compute_descriptors`` + le vrai ``Report.run()`` : seul ``num_stattest="ks"`` (le
    fix de ce correctif) doit permettre ``descriptors_drifted is False`` ici.
    """
    import src.experiments.drift as drift_module

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    def _fake_get_model(model_name, client, alias="champion"):
        return (lambda img: img.float() / 255.0), object(), None, "run123"

    monkeypatch.setattr(drift_module, "get_model_from_mlflow", _fake_get_model)

    current_paths = _write_images(
        tmp_path / "current", 20, color=(200, 200, 200), seed=1, noise_std=40
    )

    n_ref = 1200  # > 1000 : déclencherait le switch Wasserstein avec les défauts Evidently
    ref_paths = _write_images(
        tmp_path / "ref", n_ref, color=(200, 200, 200), seed=2, noise_std=40
    )
    descriptors_ref = compute_descriptors_for_test(ref_paths)
    assert len(descriptors_ref) > 1000

    def _fake_required(run_id, filename):
        assert filename == "descriptors"
        return descriptors_ref

    def _fake_optional(run_id, filename):
        return None

    monkeypatch.setattr(drift_module, "_load_required_reference", _fake_required)
    monkeypatch.setattr(drift_module, "_load_optional_reference", _fake_optional)

    result = run_drift_report("blood_cells-type", current_paths, output=tmp_path / "out")
    assert result.descriptors_drifted is False


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
