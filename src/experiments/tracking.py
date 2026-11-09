"""Adaptateur MLflow : journalise un run (ou une grille) sur le serveur MLflow/MinIO.

Activé uniquement si ``MLFLOW_TRACKING_URI`` est renseigné dans l'environnement
(convention native MLflow) ; sinon ``log_run``/``grid_run`` sont des no-op. Les
autres réglages de connexion (``MLFLOW_S3_ENDPOINT_URL``, ``AWS_ACCESS_KEY_ID``,
``AWS_SECRET_ACCESS_KEY``) sont lus directement par mlflow/boto3 — rien à faire
ici. Volontairement hors de ``ExperimentConfig`` : l'activation du tracking ne
doit pas changer le hash d'identité d'un run (``artifacts._config_hash``).

Toute erreur mlflow (serveur injoignable, etc.) est convertie en avertissement :
un run d'entraînement ne doit jamais échouer parce que le tracking est en panne.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from typing import TYPE_CHECKING

# `mlflow.pytorch` est un `LazyLoader` tant qu'il n'a pas été résolu, et le résoudre via un simple
# `getattr` (ex. dans un `monkeypatch.setattr` de test) remplace l'attribut `mlflow.pytorch` par le
# vrai module *sans* toucher la référence déjà capturée -- un patch posé sur l'ancienne référence
# serait alors silencieusement ignoré. Importer le sous-module explicitement ci-dessous force sa
# résolution une fois pour toutes, tôt (à l'import de ce module).
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from mlflow.models import ModelSignature
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.types import Schema, TensorSpec
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT

from src.config import ExperimentConfig
from src.experiments.artifacts import flatten
from src.features.transform import build_transform
from src.tasks import TASKS

# `_reinhard_from_stats` est importé localement dans `get_model_from_mlflow` (pas ici en tête de
# module) pour éviter un import circulaire : `inference.py` importe `runner.py`, qui importe
# `tracking.py` (ce module) pour `log_run` -- un import en tête de fichier créerait un cycle
# `tracking -> inference -> runner -> tracking` qui casse à l'import.

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch import nn

    from src.experiments.artifacts import RunResult


def enabled() -> bool:
    """Vrai si un serveur de tracking MLflow est configuré."""
    en = os.environ.get("MLFLOW_TRACKING_URI")
    print(f"Tracking enabled: {en or False}")
    return bool(en)


def _experiment_name(task: str) -> str:
    return f"blood_cells-{task}"


def _model_name(config: ExperimentConfig) -> str:
    return f"{config.model.transfer.backbone if config.model.name == 'transfer' else config.model.name}-{config.task}"


def _git_tags() -> dict:
    """Tag ``mlflow.source.git.commit`` depuis ``GIT_COMMIT`` (image Docker buildée avec cet ARG).

    Ne fait rien si l'env var est absente : en local, mlflow détecte déjà le
    commit via le ``.git`` réel du repo. Dans l'image, seul ``.venv`` est copié
    (pas de ``.git``, pas de binaire ``git``), donc cette détection automatique
    échoue toujours — il faut graver le commit au moment du build.
    """
    commit = os.environ.get("GIT_COMMIT")
    return {MLFLOW_GIT_COMMIT: commit} if commit else {}


def _step_metrics(history: dict) -> Iterator[tuple[int, dict]]:
    """Dévide un historique ``{"train_loss": [...], ...}`` en métriques par epoch."""
    length = max((len(v) for v in history.values() if isinstance(v, list)), default=0)
    for step in range(length):
        metrics = {
            key: values[step]
            for key, values in history.items()
            if isinstance(values, list)
            and step < len(values)
            and isinstance(values[step], (int, float))
        }
        if metrics:
            yield step, metrics


def _final_metrics(result: RunResult) -> dict:
    final: dict = {}
    final.update(
        {k: v for k, v in (result.test_metrics or {}).items() if isinstance(v, (int, float))}
    )
    final.update(
        {k: v for k, v in (result.val_metrics or {}).items() if isinstance(v, (int, float))}
    )
    final["duration_s"] = result.duration_s
    if result.n_trainable_params is not None:
        final["n_trainable_params"] = result.n_trainable_params
    return final


def get_champion_metric(client: mlflow.MlflowClient, model_name: str, metric: str):
    """Cherche le model model_name avec l'alias champion et retourne sa version et sa metric."""
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
    except mlflow.MlflowException as e:
        if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            return None, None  # modèle ou alias inexistant : premier run
        if e.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE):
            return None, None
        raise
    if not mv.run_id:
        return mv, None
    run = client.get_run(mv.run_id)
    return mv, run.data.metrics.get(metric)


def register_model(result: RunResult, model_info, metric: str = "test_f1_macro") -> bool:
    """Compare la metric du champion et du run, et définit un alias challenger si le modèle est meilleur."""
    model_name = _model_name(result.config)
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(model_name, tags={"experience": "blood_cells"})
    except mlflow.MlflowException as e:
        if e.error_code != ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
            raise
    champion_mv, champion_metric = get_champion_metric(client, model_name, metric)
    mlflow.set_tags(
        {
            "champion_metric": champion_metric,
            "champion_version": champion_mv.version if champion_mv else None,
        }
    )
    current_metric = result.test_metrics.get(metric, 0)
    if champion_metric is None or current_metric > champion_metric + 0.01:
        new_mv = mlflow.register_model(model_info.model_uri, model_name)
        client.set_registered_model_alias(model_name, "challenger", new_mv.version)
        mlflow.set_tag("promoted", "true")
    else:
        mlflow.set_tag("promoted", "false")
    return False


def get_model_from_mlflow(model_name: str, client: mlflow.MlflowClient, alias: str = "champion"):
    """Résout l'alias (ex. ``champion``) en ``(transform, model, task, run_id)``.

    Raises:
        ValueError: si le modèle ou l'alias n'existe pas dans le registre.
    """
    # Import local (pas en tête de module) : voir le commentaire sur les imports plus haut.
    from src.experiments.inference import _reinhard_from_stats

    try:
        mlflow_model = client.get_model_version_by_alias(model_name, alias)
    except mlflow.MlflowException as e:
        if e.error_code in (
            ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
            ErrorCode.Name(INVALID_PARAMETER_VALUE),
        ):
            raise ValueError(f"modèle {model_name!r} ou alias {alias!r} introuvable") from e
        raise
    model = mlflow.pytorch.load_model(mlflow_model.source)
    if not isinstance(model, torch.nn.Module):
        # mlflow>=3 sérialise par défaut en "pt2" (torch.export) hors Databricks : load_model
        # renvoie alors un torch.export.ExportedProgram, qui n'a ni .eval()/.train() ni
        # .features() et plante loin d'ici avec une trace torch illisible. log_run force déjà
        # serialization_format="pickle" pour les nouveaux runs ; un champion antérieur à ce
        # correctif doit être ré-entraîné ou re-loggé.
        raise ValueError(  # noqa: TRY004 -- ValueError, pas TypeError : server.py ne catche que
            # ValueError pour convertir en bentoml.exceptions.InvalidArgument (cf. plus bas,
            # garde-fou .features()), même contrat d'erreur pour les deux garde-fous.
            f"le modèle {model_name!r} (alias {alias!r}) a été chargé comme "
            f"{type(model).__name__!r} au lieu d'un torch.nn.Module (probablement loggé au "
            "format mlflow pt2/torch.export) — il faut le ré-entraîner ou le re-logger."
        )
    if not hasattr(model, "features"):
        # Un modèle pré-refactor .features()/.classifier() (multi_archi/baseline/dataset)
        # charge sans erreur (pickle), mais plante dès le premier forward() : sa classe
        # (résolue au unpickle, pas gelée dans le pickle) appelle self.features(...), un
        # attribut d'instance que ce vieux pickle n'a jamais eu. Voir spec §6.
        raise ValueError(
            f"le modèle {model_name!r} (alias {alias!r}) n'expose pas .features() — "
            "c'est un champion antérieur au refactor features/classifier "
            "(multi_archi/baseline/dataset) : son forward() plantera immédiatement "
            "(AttributeError/TypeError). Il faut le ré-entraîner ou le re-logger "
            "pour regagner un modèle utilisable (cf. spec §6, Limitations connues)."
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


def log_run(result: RunResult, run_dir: Path, model: nn.Module, x: np.ndarray) -> None:
    """Journalise un run terminé : params (config), métriques, artefacts (``run_dir`` entier).

    No-op si le tracking n'est pas activé. Imbriqué sous le run parent d'une
    grille si ``grid_run`` est actif (cf. ``mlflow.active_run()``).
    """
    if not enabled():
        return
    try:
        mlflow.set_experiment(_experiment_name(result.config.task))
        with mlflow.start_run(
            run_name=result.name, nested=mlflow.active_run() is not None
        ) as active_run:
            result.mlflow_run_id = active_run.info.run_id
            mlflow.set_tags({"config_hash": run_dir.name, "status": result.status, **_git_tags()})
            mlflow.log_params(flatten(result.config.model_dump(mode="json")))
            history = result.histories[0] if result.histories else {}
            for step, metrics in _step_metrics(history):
                mlflow.log_metrics(metrics, step=step)
            mlflow.log_metrics(_final_metrics(result))
            mlflow.log_artifacts(str(run_dir))
            if result.config.output.save_checkpoint:
                signature = ModelSignature(
                    inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, *x.shape[1:]))])
                )
                # serialization_format="pickle" explicite : mlflow>=3 bascule par défaut sur
                # "pt2" (torch.export.save) hors Databricks, ce qui recharge un
                # torch.export.ExportedProgram (pas un nn.Module) côté load_model -- objet qui
                # ne supporte ni .eval()/.train() ni l'appel à .features() dont dépendent le
                # service d'inférence et la référence de drift (voir src/experiments/drift_reference.py).
                model_info = mlflow.pytorch.log_model(
                    model,
                    input_example=x,
                    signature=signature,
                    name="model",
                    serialization_format="pickle",
                )
                register_model(result, model_info)
    except Exception as exc:  # noqa: BLE001 -- le tracking ne doit jamais faire échouer un run
        warnings.warn(f"mlflow indisponible, run non tracké : {exc}", stacklevel=2)


@contextlib.contextmanager
def grid_run(name: str, task: str) -> Iterator[None]:
    """Ouvre un run MLflow parent pour toute une grille ; chaque combo s'y imbrique.

    No-op si le tracking n'est pas activé. Si l'ouverture du run échoue
    (serveur injoignable), avertit et laisse la grille s'exécuter sans
    tracking plutôt que de la faire échouer.
    """
    if not enabled():
        yield
        return
    try:
        mlflow.set_experiment(_experiment_name(task))
        active = mlflow.start_run(run_name=name)
        active.__enter__()
        if git_tags := _git_tags():
            mlflow.set_tags(git_tags)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"mlflow indisponible (grid), runs non trackés : {exc}", stacklevel=2)
        yield
        return
    try:
        yield
    finally:
        active.__exit__(None, None, None)
