"""Décision de promotion champion/challenger, avec métrique interchangeable.

`decide_promotion` compare deux dicts de scores déjà réduits
(`{"socle": float, "recent": float}`) et ne branche jamais sur la valeur de
`metric` : le choix de la métrique ne concerne que la façon dont ces scores
sont calculés en amont (voir PROMOTION_METRICS, utilisé par une tâche
ultérieure), jamais la logique de comparaison elle-même.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
import mlflow.pytorch  # force la résolution du LazyLoader, voir tracking.py
import torch
import yaml

from src.config import DataConfig, ExperimentConfig, TrainConfig
from src.experiments import tracking
from src.features.dataframe import create_dataframe
from src.features.dataset import BloodCellDataset, DatasetTransform
from src.features.transform import build_transform
from src.models.metrics import build_metrics
from src.tasks import TASKS
from src.training.trainer import Trainer

if TYPE_CHECKING:
    from torch import nn

    from src.tasks import TaskSpec

PROMOTION_METRICS: dict[str, str] = {"f1": "test_f1_macro"}
DEFAULT_PROMOTION_METRIC = "f1"


def decide_promotion(
    champion_scores: dict[str, float],
    challenger_scores: dict[str, float],
    metric: str = DEFAULT_PROMOTION_METRIC,
    regression_margin: float = 0.02,
) -> bool:
    """Décide si le challenger doit remplacer le champion.

    Le challenger est promu s'il ne régresse pas sur le socle (au-delà de
    `regression_margin`) et qu'il est au moins aussi bon sur le jeu récent.
    `metric` n'est pas utilisé ici : il ne fait que documenter quelle
    métrique a servi à produire `champion_scores`/`challenger_scores` en
    amont.
    """
    no_regression = (
        challenger_scores["socle"] >= champion_scores["socle"] - regression_margin
    )
    better_on_recent = challenger_scores["recent"] >= champion_scores["recent"]
    return no_regression and better_on_recent


def evaluate_on_eval_sets(
    model: nn.Module,
    transform,
    task: TaskSpec,
    socle_key: str,
    recent_key: str,
    metric: str = DEFAULT_PROMOTION_METRIC,
    device: str = "cpu",
) -> dict[str, float]:
    """Évalue ``model`` sur le jeu socle et le jeu récent, renvoie ``{"socle": ..., "recent": ...}``.

    Même forme d'appel que l'évaluation finale de ``promote.py::_train``
    (``trainer.evaluate(test_ds, metrics=build_metrics(task, for_test=True, ...))``).
    ``criterion``/``optimizer`` ne servent qu'à satisfaire ``Trainer.__init__`` :
    aucun entraînement n'a lieu ici, seule ``evaluate()`` est appelée.
    """
    criterion = (
        torch.nn.BCEWithLogitsLoss() if task.kind == "binary" else torch.nn.CrossEntropyLoss()
    )
    trainer = Trainer(
        model,
        TrainConfig(),
        criterion,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        device=device,
    )
    metric_key = PROMOTION_METRICS[metric]
    scores: dict[str, float] = {}
    for label, key in (("socle", socle_key), ("recent", recent_key)):
        df = create_dataframe(DataConfig(datasets=[key]), task)
        ds = DatasetTransform(BloodCellDataset(df, task), transform)
        scalars = trainer.evaluate(ds, metrics=build_metrics(task, for_test=True, device=device))
        scores[label] = scalars[metric_key]
    return scores


def _find_registered_model_version(client, model_name: str, run_id: str):
    """``ModelVersion`` de ``model_name`` enregistrée pour ``run_id``, ou ``None`` si absente.

    Filtre sur ``name`` ET ``run_id`` -- un ``run_id`` seul ne suffit pas à
    identifier une version sans risque de collision avec un autre modèle
    enregistré pour le même run. S'il existe plusieurs versions pour ce couple
    (ex. ré-enregistrement), retient la plus récente (le plus grand numéro de
    version) plutôt qu'une correspondance arbitraire dépendant de l'ordre de
    recherche.
    """
    versions = client.search_model_versions(f"name='{model_name}' AND run_id='{run_id}'")
    matches = [mv for mv in versions if mv.name == model_name and mv.run_id == run_id]
    if not matches:
        return None
    return max(matches, key=lambda mv: int(mv.version))


def _load_model_version(model_name: str, mlflow_model) -> tuple:
    """Charge une version de registre déjà résolue en ``(transform, model, task, run_id)``.

    Duplique (volontairement -- voir la tâche : ``tracking.py`` n'est pas
    modifiable ici) la portion de ``tracking.get_model_from_mlflow`` qui
    s'exécute une fois l'alias/le run_id résolu en version de modèle.
    """
    # Import local (pas en tête de module) : voir le commentaire sur les imports
    # circulaires dans tracking.py -- même raison ici.
    from src.experiments.inference import _reinhard_from_stats

    model = mlflow.pytorch.load_model(mlflow_model.source)
    if not isinstance(model, torch.nn.Module):
        raise ValueError(  # noqa: TRY004 -- ValueError, pas TypeError : même contrat d'erreur
            # que tracking.get_model_from_mlflow pour ce garde-fou (cf. son commentaire).
            f"le modèle {model_name!r} (version {mlflow_model.version!r}) a été chargé comme "
            f"{type(model).__name__!r} au lieu d'un torch.nn.Module (probablement loggé au "
            "format mlflow pt2/torch.export) -- il faut le ré-entraîner ou le re-logger."
        )
    if not hasattr(model, "features"):
        raise ValueError(
            f"le modèle {model_name!r} (version {mlflow_model.version!r}) n'expose pas "
            ".features() -- champion antérieur au refactor features/classifier, "
            "à ré-entraîner ou re-logger."
        )
    run_uri = f"runs:/{mlflow_model.run_id}"
    config_data = mlflow.artifacts.load_text(f"{run_uri}/config.yaml")
    config = ExperimentConfig.model_validate(yaml.safe_load(config_data))
    preprocessing = mlflow.artifacts.load_dict(f"{run_uri}/preprocessing.json")
    normalizer = (
        _reinhard_from_stats(preprocessing["reinhard"]) if preprocessing.get("reinhard") else None
    )
    transform = build_transform(
        config.transform,
        training=False,
        reinhard_normalizer=normalizer,
        dataset_mean=torch.tensor(preprocessing["dataset_mean"]),
        dataset_std=torch.tensor(preprocessing["dataset_std"]),
    )
    return transform, model, TASKS.get(config.task), mlflow_model.run_id


def get_model_by_run_id(model_name: str, run_id: str, client, model_version=None) -> tuple:
    """Résout ``run_id`` en ``(transform, model, task, run_id)``.

    Même forme de retour que ``tracking.get_model_from_mlflow``, mais
    résolution directe par ``run_id`` plutôt que par alias -- un challenger
    issu d'un réentraînement n'a pas encore d'alias dans le registre.

    ``model_version`` : ``ModelVersion`` déjà résolue (évite une recherche
    redondante si l'appelant l'a déjà faite -- ex. ``promote_if_better``, qui
    doit de toute façon résoudre/enregistrer la version pour l'alias) ; sinon
    résolue ici via ``client.search_model_versions`` (filtré sur ``name`` ET
    ``run_id``, voir ``_find_registered_model_version``).

    Raises:
        ValueError: si ``model_version`` est absent et qu'aucune version de
            ``model_name`` n'est enregistrée pour ce ``run_id``, ou si le
            modèle chargé n'est pas un modèle utilisable (mêmes garde-fous que
            ``tracking.get_model_from_mlflow``).
    """
    mlflow_model = model_version or _find_registered_model_version(client, model_name, run_id)
    if mlflow_model is None:
        raise ValueError(f"aucune version de {model_name!r} enregistrée pour run_id={run_id!r}")
    return _load_model_version(model_name, mlflow_model)


def promote_if_better(
    model_name: str,
    challenger_run_id: str,
    socle_key: str,
    recent_key: str,
    metric: str = DEFAULT_PROMOTION_METRIC,
    client=None,
    device: str = "cpu",
    regression_margin: float = 0.02,
) -> bool:
    """Évalue le challenger face au champion et promeut si ``decide_promotion`` l'accepte.

    S'il n'y a pas encore de champion (premier réentraînement sur drift), le
    challenger est promu sans évaluation -- rien à comparer (même convention
    que ``tracking.get_champion_metric``, qui renvoie ``(None, None)`` dans ce cas).

    La version de registre du challenger est résolue UNE SEULE FOIS, ici, et
    réutilisée à la fois pour le chargement (``get_model_by_run_id``) et pour
    l'alias (``set_registered_model_alias``) -- deux recherches successives
    pourraient en théorie retomber sur des versions différentes du même run
    (ré-enregistrement concurrent). ``tracking.register_model`` ne crée une
    version que si le run franchit son propre gate F1 (standard split),
    indépendant du socle/jeu récent évalués ici : un challenger de
    réentraînement sur drift peut donc n'avoir jamais été enregistré malgré
    de bonnes performances sur le jeu récent -- on l'enregistre nous-mêmes
    dans ce cas plutôt que d'échouer.

    Returns:
        ``True`` si le challenger a été promu (alias ``champion`` déplacé),
        ``False`` sinon.
    """
    client = client or mlflow.MlflowClient()

    challenger_mv = _find_registered_model_version(client, model_name, challenger_run_id)
    if challenger_mv is None:
        challenger_mv = mlflow.register_model(f"runs:/{challenger_run_id}/model", model_name)

    champion_mv, _ = tracking.get_champion_metric(client, model_name, PROMOTION_METRICS[metric])
    if champion_mv is None:
        client.set_registered_model_alias(model_name, "champion", challenger_mv.version)
        return True

    champion_transform, champion_model, champion_task, _ = tracking.get_model_from_mlflow(
        model_name, client, "champion"
    )
    challenger_transform, challenger_model, challenger_task, _ = get_model_by_run_id(
        model_name, challenger_run_id, client, model_version=challenger_mv
    )

    # Garde-fou (bundle 6, revue finale -- déjà noté comme amélioration optionnelle dans
    # le plan d'origine) : les deux modèles comparés ci-dessous viennent toujours de
    # ``configs/densenet_data_drift.yaml`` dans ce pipeline, donc inatteignable en
    # pratique aujourd'hui -- mais si ce code est un jour réutilisé pour une autre
    # famille de modèles/tâche, comparer des scores de tâches différentes (ex. "type"
    # vs "criticality") serait silencieusement incorrect sans ce garde-fou explicite.
    if champion_task.name != challenger_task.name:
        raise ValueError(
            f"champion (task={champion_task.name!r}) et challenger (task="
            f"{challenger_task.name!r}) ne sont pas comparables : tâches différentes."
        )

    champion_scores = evaluate_on_eval_sets(
        champion_model, champion_transform, champion_task, socle_key, recent_key, metric, device=device
    )
    challenger_scores = evaluate_on_eval_sets(
        challenger_model, challenger_transform, challenger_task, socle_key, recent_key, metric, device=device
    )

    decision = decide_promotion(champion_scores, challenger_scores, metric, regression_margin=regression_margin)
    if decision:
        client.set_registered_model_alias(model_name, "champion", challenger_mv.version)
    return decision
