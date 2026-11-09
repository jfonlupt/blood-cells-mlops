"""Rapport de data drift (Evidently) entre la référence d'un run champion et de nouvelles images.

Ne relit jamais les images brutes d'entraînement : seule la référence
compacte (``drift_reference/descriptors.parquet`` + ``embeddings.parquet``
optionnel) loggée comme artefact MLflow du run champion est téléchargée.

Point d'entrée unique, partagé par le CLI (``run_drift_report``) et par le
service BentoML (``server.py`` : ``load_champion_reference`` mis en cache par
modèle, puis ``build_current_dataset``/``compute_drift_report`` par requête) --
évite d'avoir deux implémentations de la construction du rapport Evidently qui
divergent au fil des correctifs.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.legacy.metrics.data_drift.embedding_drift_methods import (
    model as embeddings_drift_method,
)
from evidently.metrics import EmbeddingsDrift
from evidently.presets import DataDriftPreset

from src.experiments.drift_reference import build_drift_reference
from src.experiments.tracking import get_model_from_mlflow

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from evidently.core.report import Snapshot
    from torch import nn

    from src.tasks import TaskSpec

EMBEDDING_GROUP = "model_embedding"
# Seuil par défaut d'evidently pour la méthode "model" (classifieur de domaine) --
# rendu explicite ici pour être relu tel quel au moment de l'interprétation du
# score (voir plus bas) plutôt que dupliqué en dur à deux endroits.
_EMBEDDINGS_DRIFT_THRESHOLD = 0.55


def _artifact_exists(run_id: str, filename: str) -> bool:
    """Vérifie si ``drift_reference/<filename>.parquet`` existe pour ``run_id``.

    Vérification faite via une liste d'artefacts (pas un try/except autour du
    téléchargement). Vérifié empiriquement (mlflow==3.17.1, backend fichier local) :
    ``client.list_artifacts(run_id, path="drift_reference")`` renvoie une liste
    vide — sans lever — aussi bien quand le sous-dossier ``drift_reference``
    n'existe pas du tout que quand seul le fichier demandé y manque. Pas besoin
    donc d'un try/except séparé pour distinguer ces deux cas : la membership
    check suffit, et une vraie panne d'infra (réseau/auth/S3) lors de cet appel
    reste non interceptée ici, donc propage normalement.
    """
    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(run_id, path="drift_reference")
    target = f"drift_reference/{filename}.parquet"
    return any(a.path == target for a in artifacts)


def _load_required_reference(run_id: str, filename: str) -> pd.DataFrame:
    """Télécharge et lit ``drift_reference/<filename>.parquet`` du run ``run_id``.

    L'existence est vérifiée via une liste d'artefacts avant le téléchargement
    (spec §3.2) : seul ``download_artifacts`` peut encore lever, et c'est alors
    une vraie panne d'infra qui doit propager telle quelle, pas être
    réinterprétée comme « référence absente ».

    Raises:
        FileNotFoundError: si l'artefact est absent (run antérieur à cette
            fonctionnalité, ou ``output.save_checkpoint=False``) — on ne peut
            rien produire sans cette référence.
    """
    if not _artifact_exists(run_id, filename):
        raise FileNotFoundError(
            f"drift_reference/{filename}.parquet absent du run {run_id} — "
            "run antérieur à cette fonctionnalité ? Impossible de produire un rapport de drift."
        )
    local_path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/drift_reference/{filename}.parquet"
    )
    return pd.read_parquet(local_path)


def _load_optional_reference(run_id: str, filename: str) -> pd.DataFrame | None:
    """Comme ``_load_required_reference``, mais renvoie ``None`` si l'artefact est absent.

    Même garde par liste d'artefacts que ``_load_required_reference`` — une
    vraie panne d'infra lors du téléchargement n'est, elle, jamais avalée.
    """
    if not _artifact_exists(run_id, filename):
        return None
    local_path = mlflow.artifacts.download_artifacts(
        f"runs:/{run_id}/drift_reference/{filename}.parquet"
    )
    return pd.read_parquet(local_path)


def _assert_same_columns(ref: pd.DataFrame, cur: pd.DataFrame, name: str) -> None:
    """Garde-fou de cohérence de schéma : les colonnes doivent matcher exactement, en ordre.

    Un décalage silencieux entre les descripteurs/embeddings calculés à
    l'entraînement et ceux recalculés à la consommation (changement de code
    entre les deux, ordre de colonnes différent, etc.) doit faire échouer
    bruyamment plutôt que produire un rapport Evidently trompeur.

    Raises:
        ValueError: si les colonnes de ``ref`` et ``cur`` diffèrent.
    """
    if list(ref.columns) != list(cur.columns):
        raise ValueError(
            f"colonnes {name} incohérentes entre la référence et les nouvelles données : "
            f"référence={list(ref.columns)!r} vs courant={list(cur.columns)!r}"
        )


@dataclass
class ChampionReference:
    """Tout ce qu'il faut, chargé une fois par modèle champion, pour comparer un lot d'images.

    Coûteux à construire (téléchargement MLflow) : le CLI (``run_drift_report``)
    en construit une par invocation, le service (``server.py``) en garde une en
    cache par nom de modèle et ne reconstruit que le ``current`` par requête
    (``build_current_dataset`` + ``compute_drift_report``).
    """

    model_name: str
    run_id: str
    transform: Callable
    model: nn.Module
    task: TaskSpec | None
    descriptors_ref: pd.DataFrame
    embeddings_ref: pd.DataFrame | None
    numerical_features: list[str]
    has_embeddings: bool
    data_definition: DataDefinition
    reference_dataset: Dataset


def load_champion_reference(
    model_name: str, client: mlflow.MlflowClient, alias: str = "champion"
) -> ChampionReference:
    """Charge le modèle champion et sa référence de drift (spec §3.2).

    Raises:
        ValueError: si le modèle ou l'alias n'existe pas dans le registre, ou si le
            modèle champion n'expose pas ``.features()`` (cf. ``get_model_from_mlflow``).
        FileNotFoundError: si le run champion n'a pas de référence de drift
            (run antérieur à cette fonctionnalité).
    """
    transform, model, task, run_id = get_model_from_mlflow(model_name, client, alias)

    descriptors_ref = _load_required_reference(run_id, "descriptors")
    embeddings_ref = _load_optional_reference(run_id, "embeddings")
    numerical_features = [c for c in descriptors_ref.columns if c != "path"]
    has_embeddings = embeddings_ref is not None

    if not has_embeddings:
        warnings.warn(
            f"section embeddings absente du rapport de drift : "
            f"drift_reference/embeddings.parquet absent du run champion {run_id} "
            "(modèle sans .features() à l'entraînement, ou run antérieur à cette "
            "fonctionnalité).",
            stacklevel=2,
        )
        reference_df = descriptors_ref
        data_definition = DataDefinition(numerical_columns=numerical_features)
    else:
        reference_df = descriptors_ref.merge(embeddings_ref, on="path")
        emb_cols = [c for c in embeddings_ref.columns if c != "path"]
        data_definition = DataDefinition(
            numerical_columns=numerical_features, embeddings={EMBEDDING_GROUP: emb_cols}
        )

    reference_dataset = Dataset.from_pandas(
        reference_df.drop(columns=["path"]), data_definition=data_definition
    )

    return ChampionReference(
        model_name=model_name,
        run_id=run_id,
        transform=transform,
        model=model,
        task=task,
        descriptors_ref=descriptors_ref,
        embeddings_ref=embeddings_ref,
        numerical_features=numerical_features,
        has_embeddings=has_embeddings,
        data_definition=data_definition,
        reference_dataset=reference_dataset,
    )


@dataclass
class CurrentBatch:
    """Lot d'images courantes, déjà aligné sur le schéma de référence d'un ``ChampionReference``."""

    dataset: Dataset
    has_embeddings: bool
    n_images: int


def build_current_dataset(
    champion: ChampionReference, image_paths: Sequence[str | Path]
) -> CurrentBatch:
    """Calcule descripteurs/embeddings de ``image_paths``, alignés sur le schéma de ``champion``.

    Raises:
        ValueError: si ``image_paths`` est vide, ou si les colonnes recalculées ne
            correspondent pas à celles de la référence (spec §3.2).
    """
    if not image_paths:
        raise ValueError("aucune image fournie — --images doit contenir au moins un fichier")

    current_ref = build_drift_reference(
        [str(p) for p in image_paths], champion.transform, champion.model, device="cpu"
    )
    descriptors_cur = current_ref.descriptors
    _assert_same_columns(champion.descriptors_ref, descriptors_cur, "descripteurs")

    embeddings_cur = current_ref.embeddings if champion.has_embeddings else None
    if champion.has_embeddings and embeddings_cur is None:
        warnings.warn(
            "section embeddings absente du rapport de drift : le modèle champion "
            "actuel n'expose pas .features() — impossible de calculer les embeddings "
            "des nouvelles images, bien que la référence en contienne.",
            stacklevel=2,
        )

    has_embeddings = champion.has_embeddings and embeddings_cur is not None
    if has_embeddings:
        _assert_same_columns(champion.embeddings_ref, embeddings_cur, "embeddings")
        current_df = descriptors_cur.merge(embeddings_cur, on="path")
    else:
        current_df = descriptors_cur

    dataset = Dataset.from_pandas(
        current_df.drop(columns=["path"]), data_definition=champion.data_definition
    )
    return CurrentBatch(dataset=dataset, has_embeddings=has_embeddings, n_images=len(current_df))


@dataclass
class DriftMetricsResult:
    """Résultat brut du calcul de drift : flags de synthèse + ``Snapshot`` evidently.

    Le ``Snapshot`` reste à la charge de l'appelant (sauvegarde HTML/JSON pour le
    CLI, ``.dict()`` direct pour le service) : ce module ne décide pas comment le
    résultat est restitué.
    """

    descriptors_drifted: bool
    embeddings_drifted: bool | None
    snapshot: Snapshot


def is_drift_detected(result: DriftMetricsResult) -> bool:
    """Vrai si descripteurs ou embeddings dérivent ; ``None`` (section absente) compte comme non dérivé."""
    return bool(result.descriptors_drifted) or bool(result.embeddings_drifted)


def compute_drift_report(
    champion: ChampionReference,
    current: CurrentBatch,
    extra_metrics: list | None = None,
) -> DriftMetricsResult:
    """Calcule le rapport de drift de ``current`` contre la référence de ``champion``.

    Args:
        champion: Référence du modèle champion (cf. ``load_champion_reference``).
        current: Lot courant déjà aligné sur son schéma (cf. ``build_current_dataset``).
        extra_metrics: Métriques evidently additionnelles, ajoutées telles quelles à la
            liste (ex. futur metric_v2 de qualité de classification dans ``/evaluate``,
            à partir des labels fournis par l'appelant — non implémenté ici, ce paramètre
            existe pour que ``server.py`` puisse le brancher sans retoucher ce module).
    """
    # num_method="ks" (Kolmogorov-Smirnov) forcé pour toutes les colonnes
    # numériques : le défaut d'Evidently bascule automatiquement sur une
    # distance de Wasserstein normée (seuil fixe 0.1) dès que la référence
    # dépasse 1000 lignes -- notre référence (200 images/classe) dépasse ce
    # seuil à partir de 5-6 classes, et ce test Wasserstein déclenche alors un
    # faux positif quasi systématique sur de petits lots courants (mesuré :
    # 100% de faux positifs à 20 images courantes, décroissant seulement au-delà
    # de ~300). KS reste valide côté petit échantillon courant et ne dépend pas
    # de la taille de la référence.
    #
    # Contrairement à evidently 0.4.x, DataDriftPreset (API "metric_v2" native
    # d'evidently>=0.7) n'inclut plus automatiquement le drift des embeddings
    # même quand column_mapping/data_definition en déclare un groupe (vérifié
    # empiriquement : ses embeddings/embeddings_drift_method ne sont pas câblés
    # dans generate_metrics()) -- il faut l'ajouter explicitement via le
    # métrique natif EmbeddingsDrift, qui délègue en interne à l'ancien
    # EmbeddingsDriftMetric (même méthode "model", même sémantique).
    has_embeddings = current.has_embeddings
    metrics: list = [DataDriftPreset(num_method="ks")]
    if has_embeddings:
        metrics.append(
            EmbeddingsDrift(
                embeddings_name=EMBEDDING_GROUP,
                drift_method=embeddings_drift_method(threshold=_EMBEDDINGS_DRIFT_THRESHOLD),
            )
        )
    metrics.extend(extra_metrics or [])

    try:
        report = Report(metrics=metrics)
        snapshot = report.run(current_data=current.dataset, reference_data=champion.reference_dataset)
    except (ValueError, np.linalg.LinAlgError):
        # EmbeddingsDrift calcule déjà drift_score/drift_detected à ce stade, mais plante
        # ensuite sur sa SEULE visualisation (evidently/legacy/metrics/data_drift/
        # embeddings_drift.py : projection t-SNE à 2D de (référence + courant) puis
        # gaussian_kde sur les coordonnées courantes). Deux façons d'échouer, observées
        # empiriquement selon la taille des échantillons : sklearn TSNE lève un
        # ValueError ("perplexity must be less than n_samples") si référence+courant
        # total reste sous ~30-31 lignes ; gaussian_kde lève un LinAlgError (covariance
        # singulière) avec moins de 3 images courantes, quelle que soit la taille de la
        # référence (2 points ne peuvent jamais définir une covariance 2D de rang plein).
        # calculate() ne renvoie rien de partiel en cas d'exception : on ne peut récupérer
        # drift_score, donc on relance sans la métrique embeddings plutôt que de perdre tout
        # le rapport (y compris la section descripteurs, elle non concernée par ce problème).
        if not has_embeddings:
            raise
        warnings.warn(
            "section embeddings absente du rapport de drift : la visualisation interne "
            f"d'evidently (projection t-SNE + KDE) échoue avec seulement {current.n_images} "
            "image(s) courante(s) -- au moins 3 images courantes sont nécessaires pour cette "
            "section. Le score de drift des descripteurs reste calculé normalement.",
            stacklevel=2,
        )
        has_embeddings = False
        report = Report(metrics=[DataDriftPreset(num_method="ks"), *(extra_metrics or [])])
        snapshot = report.run(current_data=current.dataset, reference_data=champion.reference_dataset)

    summary = snapshot.dict()

    # L'API native (contrairement à l'ancienne) ne renvoie pas de booléen "détecté"
    # tout fait : on le recalcule depuis la valeur mesurée et le seuil/drift_share
    # relus depuis le `config` de la métrique elle-même (source unique, pas de
    # constante dupliquée).
    drifted_columns_entry = next(
        m
        for m in summary["metrics"]
        if m["config"]["type"] == "evidently:metric_v2:DriftedColumnsCount"
    )
    descriptors_drifted = (
        drifted_columns_entry["value"]["share"] > drifted_columns_entry["config"]["drift_share"]
    )

    embeddings_drifted = None
    if has_embeddings:
        embeddings_entry = next(
            m
            for m in summary["metrics"]
            if m["config"]["type"] == "evidently:metric_v2:EmbeddingsDrift"
        )
        embeddings_drifted = (
            embeddings_entry["value"] > embeddings_entry["config"]["drift_method"]["threshold"]
        )

    return DriftMetricsResult(
        descriptors_drifted=descriptors_drifted,
        embeddings_drifted=embeddings_drifted,
        snapshot=snapshot,
    )


@dataclass
class DriftReportResult:
    """Résultat d'un rapport de drift CLI : flags de synthèse + chemin du rapport HTML."""

    descriptors_drifted: bool
    embeddings_drifted: bool | None
    report_path: Path


def run_drift_report(
    model_name: str,
    image_paths: Sequence[str | Path],
    alias: str = "champion",
    output: Path = Path("drift_report"),
) -> DriftReportResult:
    """Compare ``image_paths`` à la référence du modèle champion, sans relire le train brut.

    Fin couche CLI au-dessus de ``load_champion_reference``/``build_current_dataset``/
    ``compute_drift_report`` (aussi utilisées par ``server.py`` /evaluate) : ne fait que
    charger la référence une fois, calculer le rapport, et écrire les artefacts fichier.

    Raises:
        ValueError: si ``image_paths`` est vide.
        FileNotFoundError: si le run champion n'a pas de référence de drift
            (run antérieur à cette fonctionnalité).
    """
    if not image_paths:
        raise ValueError("aucune image fournie — --images doit contenir au moins un fichier")

    champion = load_champion_reference(model_name, mlflow.MlflowClient(), alias)
    current = build_current_dataset(champion, image_paths)
    result = compute_drift_report(champion, current)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "drift_report.html"
    result.snapshot.save_html(str(report_path))
    summary = result.snapshot.dict()
    (output / "drift_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    return DriftReportResult(
        descriptors_drifted=result.descriptors_drifted,
        embeddings_drifted=result.embeddings_drifted,
        report_path=report_path,
    )
