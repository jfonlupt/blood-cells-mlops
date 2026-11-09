"""Déclenchement de réentraînement piloté par le drift détecté en production.

Module autonome, sans dépendance à Airflow (non installé dans cet
environnement ; Airflow tourne dans une image Docker séparée). Il sera
importé par un DAG Airflow (tâche ultérieure), mais reste ici un simple
module Python testable par pytest.
"""

from __future__ import annotations

import statistics
import subprocess
import time
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from pathlib import Path


def query_drift_share(
    prometheus_url: str,
    model: str,
    window_minutes: int = 15,
    step_seconds: int = 60,
    now: float | None = None,
) -> float:
    """Part des échantillons récents de ``model_drift_detected`` égaux à 1.

    Interroge l'API HTTP de Prometheus (``query_range``) sur la fenêtre
    ``[now - window_minutes, now]`` pour le modèle donné, et renvoie la
    fraction des valeurs échantillonnées qui valent 1.

    Args:
        prometheus_url: URL de base du serveur Prometheus (sans suffixe).
        model: Valeur du label ``model`` à filtrer.
        window_minutes: Largeur de la fenêtre temporelle interrogée.
        step_seconds: Pas d'échantillonnage demandé à Prometheus.
        now: Horodatage de fin de fenêtre (epoch, secondes) ; par défaut
            ``time.time()``.

    Returns:
        La fraction (entre 0.0 et 1.0) d'échantillons à 1, toutes séries
        confondues. Renvoie ``0.0`` quand Prometheus ne rapporte aucune
        série correspondante (absence de données traitée comme absence de
        signal de drift, jamais comme une erreur).
    """
    if now is None:
        now = time.time()

    resp = requests.get(
        f"{prometheus_url}/api/v1/query_range",
        params={
            "query": f'model_drift_detected{{model="{model}"}}',
            "start": now - window_minutes * 60,
            "end": now,
            "step": step_seconds,
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    result = payload["data"]["result"]
    if not result:
        return 0.0

    values = [value for series in result for _, value in series["values"]]
    return statistics.fmean(float(value) for value in values)


def is_drift_sustained(share: float, min_share: float = 0.8) -> bool:
    """``True`` si la part de drift détecté atteint le seuil ``min_share``."""
    return share >= min_share


def latest_commit_touching(repo_dir: Path, relative_path: str) -> str | None:
    """SHA du commit le plus récent ayant modifié ``relative_path``.

    Renvoie ``None`` si ``relative_path`` n'a aucun historique (jamais
    commité, ou dépôt sans aucun commit — branche "unborn"). Toute autre
    erreur git (ex. ``repo_dir`` n'est pas un dépôt git) est relevée telle
    quelle, car ce n'est pas la même situation : il n'y a alors aucune
    garantie qu'il n'existe pas d'historique, seulement que git n'a pas pu
    répondre.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "log", "-1", "--format=%H", "--", relative_path],  # noqa: S607
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if "does not have any commits yet" in result.stderr:
            return None
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )

    commit = result.stdout.strip()
    return commit or None


def latest_training_commit(
    client, experiment_name: str, run_name: str | None = None
) -> str | None:
    """SHA du commit git tagué sur le run d'entraînement le plus récent.

    Lit le tag ``mlflow.source.git.commit`` (déjà écrit par
    ``tracking._git_tags``/``log_run``) sur le dernier run de
    ``experiment_name``. Renvoie ``None`` si l'expérience n'a aucun run, ou
    si le run le plus récent n'a pas ce tag.

    ``run_name`` : si fourni, restreint la recherche aux runs nommés
    ``run_name`` (tag ``mlflow.runName``) et au statut ``FINISHED``. Sans ce
    filtre, ``experiment_name`` (ex. ``"blood_cells-type"``) est partagé par
    les grid sweeps, d'autres backbones, des runs locaux de développeur et
    des runs échoués -- une recherche non scopée peut donc renvoyer un run
    sans rapport avec ce pipeline de réentraînement sur drift, et fausser à
    la fois le guard "déjà entraîné" et "il y a de nouvelles données"
    (finding 3 de la revue finale). Paramètre optionnel pour ne pas casser
    les appelants existants qui n'ont pas besoin de ce scoping.
    """
    # Correction opportuniste (hors liste des findings, repérée en touchant
    # cette fonction pour le finding 3) : ``MlflowClient.search_runs`` (la
    # vraie classe, pas les doubles de test ci-contre) n'accepte PAS de
    # paramètre ``experiment_names`` -- seulement ``experiment_ids`` (vérifié
    # empiriquement contre mlflow-skinny 3.17.1 installé : passer
    # ``experiment_names`` lève un ``TypeError``). Seule la fonction fluent
    # ``mlflow.search_runs`` (module-level) accepte des noms. Il faut donc
    # résoudre l'expérience par son nom avant de chercher les runs.
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    filter_string = None
    if run_name is not None:
        filter_string = (
            f"tags.mlflow.runName = '{run_name}' AND attributes.status = 'FINISHED'"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None

    return runs[0].data.tags.get("mlflow.source.git.commit")


def has_new_data_since_last_training(
    repo_dir: Path,
    relative_path: str,
    client,
    experiment_name: str,
    run_name: str | None = None,
) -> bool:
    """``True`` si des données plus récentes que le dernier entraînement existent.

    Compare, via l'ascendance git, le dernier commit ayant touché
    ``relative_path`` au commit tagué sur le dernier run d'entraînement.
    Si aucun run d'entraînement n'est tagué (ou n'existe), renvoie
    ``True`` par défaut (hypothèse la plus sûre : il y a toujours quelque
    chose de nouveau à utiliser).

    ``run_name`` : transmis tel quel à ``latest_training_commit`` pour scoper
    la recherche du dernier run d'entraînement (finding 3, revue finale) --
    voir sa docstring.
    """
    training_commit = latest_training_commit(client, experiment_name, run_name=run_name)
    if training_commit is None:
        return True

    data_commit = latest_commit_touching(repo_dir, relative_path)
    if data_commit is None:
        # Aucun commit n'a jamais touché ce chemin : rien n'a été capturé du
        # tout. Contrairement au cas "pas encore de run d'entraînement" (où
        # l'hypothèse la plus sûre est "il y a du nouveau"), il n'y a ici
        # aucun univers plausible où l'absence totale de données committées
        # compterait comme "nouvelles données à exploiter".
        return False

    result = subprocess.run(  # noqa: S603
        ["git", "merge-base", "--is-ancestor", data_commit, training_commit],  # noqa: S607
        cwd=repo_dir,
        check=False,
    )
    return result.returncode != 0
