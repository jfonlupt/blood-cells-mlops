"""DAG Airflow : réentraînement déclenché par le drift détecté en production.

Pur wiring : toute la logique (requête du drift, détection de nouvelles
données, décision de promotion champion/challenger) vit dans
``src/experiments/retrain_trigger.py`` et ``src/experiments/promotion.py``
-- modules testés par pytest, sans dépendance à Airflow. Ce fichier se
limite à appeler ces fonctions avec leurs vraies signatures et à les
enchaîner dans un DAG. Il n'est pas (et ne peut pas être) importé par la
suite pytest de ce repo : ``apache-airflow`` ne fait pas partie de
l'environnement ``uv`` du projet, seulement de l'image ``apache/airflow``
utilisée par ``docker-compose.yaml``/``make airflow``.

Les imports de ``src.experiments.promotion``/``src.experiments.retrain_trigger``
et de ``mlflow`` sont volontairement faits DANS le corps des tâches (pas en
tête de module) : ``promotion.py`` importe ``torch``/``mlflow.pytorch``/
``src.training.trainer``/etc., et un DAG Airflow est re-parsé par le
scheduler environ toutes les 30s -- un import module-level qui échoue
ferait disparaître tout le DAG de l'UI (erreur de parsing), alors qu'un
import en échec dans le corps d'une tâche n'échoue que l'exécution de cette
tâche-là.

Prérequis infra PAS couverts par ce fichier (changements à
``docker-compose.yaml``, hors scope de cette tâche -- nécessaires avant que
la vérification manuelle de l'étape 2 du plan puisse réussir) :

- ``apache-airflow-providers-docker`` doit être installé dans l'image
  Airflow (ex. via ``_PIP_ADDITIONAL_REQUIREMENTS``, actuellement
  commenté dans ``docker-compose.yaml``) pour que ``DockerOperator``
  (et ``docker.types.Mount``, dépendance du SDK docker qu'il embarque)
  soient importables ;
- ``/var/run/docker.sock`` doit être monté dans les conteneurs
  ``airflow-scheduler``/``airflow-worker`` pour que ``DockerOperator``
  puisse parler au démon Docker hôte ;
- le dépôt git complet (avec ``.git``) doit être monté dans ces mêmes
  conteneurs (ex. ``.:/opt/airflow/repo:ro``) pour que
  ``has_new_data_since_last_training``/``latest_commit_touching``
  (``git log``/``git merge-base`` en sous-processus) fonctionnent -- seuls
  ``./dags``, ``./logs``, ``./plugins`` sont montés aujourd'hui ;
- le conteneur d'entraînement lancé par ``DockerOperator`` est un
  conteneur Docker autonome, pas un service du ``docker-compose.yaml`` : il
  ne reçoit donc ni le réseau ni les variables d'environnement
  (``MLFLOW_TRACKING_URI``, identifiants MinIO/S3, ``DATA_DIR``) que
  ``docker-compose.yaml`` injecte au service ``blood_cells_training`` --
  d'où leur reprise explicite ci-dessous via ``os.environ`` (donc à fournir
  aux conteneurs Airflow eux-mêmes) et un ``network_mode`` à pointer vers le
  réseau du projet compose (``docker network ls`` pour le nom exact, en
  général ``<nom_projet>_default``) ;
- la tâche ``promote`` tourne, elle, DANS le conteneur
  ``airflow-worker`` (pas dans un conteneur Docker séparé) : ce conteneur a
  donc lui-même besoin de ``torch``/``mlflow``/les dépendances ML du projet
  installées, de ``DATA_DIR`` pointant vers un jeu de données monté, et de
  ``MLFLOW_TRACKING_URI``/des identifiants S3 valides dans son propre
  environnement -- puisque ``promote_if_better`` y construit son
  ``mlflow.MlflowClient()`` et y évalue réellement les modèles champion/
  challenger (``evaluate_on_eval_sets``) ;
- de même, ``check_drift_and_new_data`` tourne dans ``airflow-worker`` et y
  a besoin de ``mlflow``/``requests`` installés et de
  ``MLFLOW_TRACKING_URI`` valide pour son ``mlflow.MlflowClient()``.
- ``docker.types.Mount`` (utilisé ci-dessous pour les montages du
  conteneur d'entraînement) attend des chemins HÔTE (résolus par le démon
  Docker via ``docker.sock``), jamais des chemins tels que vus depuis le
  conteneur Airflow -- cf. ``TRAIN_HOST_*_DIR`` ci-dessous, dont les valeurs
  par défaut sont des placeholders non fonctionnels à surcharger par
  environnement.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import dag, task
from docker.types import Mount

# -- Constantes de déploiement -------------------------------------------
# Valeurs par défaut pour un lancement local type ``make airflow`` ; à
# surcharger via l'environnement des conteneurs airflow-scheduler/-worker
# pour un déploiement réel plutôt qu'en les modifiant ici.
PROMETHEUS_URL = os.environ.get("RETRAIN_PROMETHEUS_URL", "http://prometheus:9090")
# "densenet121-type" = tracking._model_name() pour configs/densenet_data_drift.yaml
# (model.transfer.backbone=densenet121, task=type) ; à garder synchronisé si ce
# fichier de config change de backbone/tâche.
MODEL_NAME = os.environ.get("RETRAIN_MODEL_NAME", "densenet121-type")
# "blood_cells-type" = tracking._experiment_name("type"), même remarque.
EXPERIMENT_NAME = os.environ.get("RETRAIN_EXPERIMENT_NAME", "blood_cells-type")
REPO_DIR = Path(os.environ.get("RETRAIN_REPO_DIR", "/opt/airflow/repo"))
# Chemin dans le conteneur d'entraînement : docker-compose.yaml monte ./configs
# sur /app/config pour le service blood_cells_training.
TRAIN_CONFIG_PATH = "/app/config/densenet_data_drift.yaml"
# Réseau docker-compose du projet, pour que le conteneur d'entraînement
# puisse résoudre les noms de service "mlflow"/"minio" -- placeholder non
# fonctionnel (cf. le prérequis infra documenté en tête de ce fichier),
# à surcharger via RETRAIN_DOCKER_NETWORK (``docker network ls`` pour le nom
# exact du réseau compose, en général ``<nom_projet>_default``).
TRAIN_NETWORK_MODE = os.environ.get("RETRAIN_DOCKER_NETWORK", "bridge")
# Chemins HÔTE (résolus par le démon Docker via docker.sock -- PAS les
# chemins tels que vus depuis le conteneur Airflow) à monter en bind dans le
# conteneur d'entraînement. Placeholders non fonctionnels : à surcharger via
# ces variables d'environnement pour un déploiement réel, même remarque que
# TRAIN_NETWORK_MODE ci-dessus.
TRAIN_HOST_CONFIGS_DIR = os.environ.get("RETRAIN_HOST_CONFIGS_DIR", "/path/to/repo/configs")
TRAIN_HOST_DATA_DIR = os.environ.get("RETRAIN_HOST_DATA_DIR", "/path/to/repo/data")
TRAIN_HOST_RESULTS_DIR = os.environ.get("RETRAIN_HOST_RESULTS_DIR", "/path/to/repo/results")

MLFLOW_RUN_ID_PATTERN = re.compile(r"MLflow run id\s*:\s*(\S+)")
# Nom du run d'entraînement poussé par ce DAG (cf. "--name" passé au conteneur
# d'entraînement dans la définition de la tâche `train` ci-dessous) -- utilisé pour
# scoper la recherche MLflow de `latest_training_commit` à ce pipeline (finding 3,
# revue finale) plutôt qu'à tout `EXPERIMENT_NAME` (partagé avec les grid sweeps,
# d'autres backbones, des runs locaux de développeur, des runs échoués...).
TRAIN_RUN_NAME = "retrain_on_drift"
# Chemin git du "pool entraînable" de données drift : la simulation du drift
# (src/helpers/data_drift.py) fait un `dvc add` sur `data/train/data_drift` et
# `data/test/data_drift`, que git ne suit donc que via les fichiers `.dvc`
# correspondants -- les dossiers d'images eux-mêmes sont gitignorés
# (`.gitignore`: `data/**` + `!data/**/*.dvc`). `data/data_drift` (chemin utilisé par
# une version antérieure de ce fichier, reprise de l'erreur du plan d'origine) n'a
# jamais été commité : `git log -- data/data_drift` ne renvoie donc jamais rien, et
# le guard "nouvelles données" verrait toujours "rien de nouveau" dès qu'un premier
# run d'entraînement existe. On surveille spécifiquement le chemin TRAIN (pas TEST) :
# un commit eval-only (`data/test/data_drift.dvc`) ne doit pas déclencher de
# réentraînement, puisque l'entraînement ne lit jamais ce jeu de données.
DRIFT_TRAIN_DVC_PATH = "data/train/data_drift.dvc"


@task.branch
def check_drift_and_new_data(ti=None) -> str:
    """``"train"`` si drift soutenu ET données plus récentes que le dernier run, sinon ``"skip"``.

    Pousse aussi le commit HEAD courant sous la clé XCom ``git_commit``, pour
    que la tâche ``train`` (``DockerOperator``, dont ``environment`` est un
    champ templaté) puisse le lire via Jinja plutôt que de le calculer en
    dehors de toute tâche -- un appel git fait au niveau module serait
    ré-exécuté à chaque cycle de parsing du DAG par le scheduler (~30s) et
    ferait planter l'analyse de tout le fichier (donc disparaître le DAG de
    l'UI) si ``REPO_DIR`` n'est pas (encore) un dépôt git, ce qui est le cas
    tant que le prérequis infra du mount du dépôt n'est pas en place.
    """
    import mlflow

    from src.experiments.retrain_trigger import (
        has_new_data_since_last_training,
        is_drift_sustained,
        latest_commit_touching,
        query_drift_share,
    )

    share = query_drift_share(PROMETHEUS_URL, MODEL_NAME)
    drift_ok = is_drift_sustained(share)
    data_ok = has_new_data_since_last_training(
        REPO_DIR,
        DRIFT_TRAIN_DVC_PATH,
        mlflow.MlflowClient(),
        EXPERIMENT_NAME,
        run_name=TRAIN_RUN_NAME,
    )

    # "unknown" si le dépôt n'a jamais été commité (mêmes conventions que
    # docker/Dockerfile : ARG GIT_COMMIT=unknown).
    git_commit = latest_commit_touching(REPO_DIR, ".") or "unknown"
    ti.xcom_push(key="git_commit", value=git_commit)

    return "train" if (drift_ok and data_ok) else "skip"


@task(retries=2)
def promote(train_output: str | list[str]) -> bool:
    """Extrait le run id MLflow de la sortie du conteneur d'entraînement et décide de la promotion.

    ``retries=2`` (mitigation, finding 5 revue finale) : le guard de
    ``has_new_data_since_last_training`` ne vérifie que "un run a bien été loggé
    pour ce commit de données", pas "ce run a bien été évalué/promu". Si
    l'entraînement réussit mais que CETTE tâche échoue ensuite (erreur
    transitoire MLflow/S3 pendant l'évaluation), le prochain passage du DAG
    verra "déjà entraîné" et sautera -- le challenger reste orphelin sans
    retry. Les retries automatiques couvrent les échecs transitoires, mais ne
    ferment pas complètement la fenêtre (un échec persistant au-delà des
    retries laisse quand même le challenger orphelin). Le correctif complet
    nécessiterait de tagger les runs post-évaluation et de faire remonter ce
    tag dans retrain_trigger.py/has_new_data_since_last_training -- changement
    de conception plus large, explicitement hors scope de ce round de fix.
    """
    from src.experiments.promotion import promote_if_better

    # xcom_all=True (DockerOperator `train`) renvoie `log_lines`, une liste de
    # str (chaque ligne de stdout du conteneur), pas une str jointe -- sans
    # cette normalisation, MLFLOW_RUN_ID_PATTERN.search() lève un TypeError
    # sur une liste (finding 2, revue finale).
    if isinstance(train_output, list):
        train_output = "\n".join(train_output)

    match = MLFLOW_RUN_ID_PATTERN.search(train_output)
    if match is None:
        raise ValueError(
            "run id MLflow introuvable dans la sortie du conteneur d'entraînement : "
            f"{train_output!r}"
        )
    run_id = match.group(1)
    if run_id == "None":
        # result.mlflow_run_id est None quand le tracking MLflow est désactivé
        # (cf. tracking.enabled()) -- le CLI imprime alors littéralement
        # "MLflow run id : None". Mieux vaut échouer ici avec un message clair
        # que de laisser promote_if_better planter plus loin en tentant de
        # charger runs:/None/model.
        raise ValueError(
            "tracking MLflow désactivé pour ce run d'entraînement (run id = None) -- "
            "impossible de promouvoir un challenger sans run id MLflow réel."
        )
    return promote_if_better(MODEL_NAME, run_id, "munich_eval", "data_drift_eval")


@dag(dag_id="retrain_on_drift", schedule="*/10 * * * *", max_active_runs=1, catchup=False)
def retrain_on_drift():
    skip = EmptyOperator(task_id="skip")

    # ATTENTION : l'image blood_cells_training:latest est construite depuis
    # docker/Dockerfile, dont l'ENTRYPOINT baked-in lance le serveur Bento
    # (uvicorn src.experiments.server:app) -- ce n'est le script
    # d'entraînement que parce que docker-compose.yaml le redéfinit au
    # niveau du service via sa propre clé `entrypoint:`. DockerOperator
    # instancie un conteneur à partir de l'IMAGE directement (pas du
    # service compose), donc l'ENTRYPOINT de l'image doit être explicitement
    # écrasé ici, sous peine de lancer le serveur Bento au lieu de
    # l'entraînement.
    train = DockerOperator(
        task_id="train",
        image="blood_cells_training:latest",
        entrypoint=["python", "-m", "src.experiments.run"],
        command=["--config", TRAIN_CONFIG_PATH, "--name", TRAIN_RUN_NAME],
        environment={
            # Champ templaté Jinja sur DockerOperator : lu à l'exécution de la
            # tâche depuis le XCom poussé par check_drift_and_new_data, jamais
            # calculé par un appel git au niveau module (cf. docstring du
            # module et de check_drift_and_new_data).
            "GIT_COMMIT": (
                "{{ ti.xcom_pull(task_ids='check_drift_and_new_data', key='git_commit') }}"
            ),
            # URIs de service (MLflow tracking, endpoint S3/MinIO) : pas des secrets en
            # eux-mêmes (ce sont des adresses, pas des identifiants) -- `environment` est
            # un champ templaté Jinja (visible dans la vue "Rendered Template" de l'UI
            # Airflow et dans le DAG sérialisé), donc on y laisse volontairement
            # uniquement ce qui n'a pas besoin d'être caché. Finding 4, revue finale.
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
            "MLFLOW_ARTIFACT_URI": os.environ.get("MLFLOW_ARTIFACT_URI", ""),
            "MLFLOW_S3_ENDPOINT_URL": os.environ.get("MLFLOW_S3_ENDPOINT_URL", ""),
            "DATA_DIR": "/app/data",
        },
        # Identifiants réels (secrets) : `private_environment` n'est PAS un champ
        # templaté -- sa valeur n'apparaît jamais dans la vue "Rendered Template" de
        # l'UI Airflow ni dans le DAG sérialisé, contrairement à `environment`
        # ci-dessus (finding 4, revue finale).
        private_environment={
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        },
        mounts=[
            Mount(source=TRAIN_HOST_CONFIGS_DIR, target="/app/config", type="bind", read_only=True),
            Mount(source=TRAIN_HOST_DATA_DIR, target="/app/data", type="bind"),
            Mount(source=TRAIN_HOST_RESULTS_DIR, target="/app/results", type="bind"),
        ],
        # Le DataLoader PyTorch (num_workers: 4 dans densenet_data_drift.yaml)
        # échange ses tenseurs entre workers via /dev/shm -- Docker limite ce
        # tmpfs à 64 Mo par défaut, bien trop petit pour des batches d'images,
        # et ça fait planter le conteneur dès le premier batch avec
        # "RuntimeError: unable to allocate shared memory(shm)... No space
        # left on device" (constaté le 2026-10-26).
        shm_size="2g",
        # Évite la tentative ratée (et bruyante dans les logs) de monter un
        # tmp dir hôte avant de retomber sur mount_tmp_dir=False : on est
        # toujours en docker-in-docker ici (DockerOperator ne voit que
        # docker.sock, pas le filesystem de l'hôte réel), donc ce fallback
        # est systématique -- autant le déclarer explicitement.
        mount_tmp_dir=False,
        network_mode=TRAIN_NETWORK_MODE,
        do_xcom_push=True,
        # Par défaut, seule la dernière ligne de stdout est poussée en XCom ;
        # xcom_all=True évite de dépendre de "MLflow run id : ..." étant
        # strictement la toute dernière ligne affichée (ex. un warning
        # mlflow/atexit imprimé après) -- MLFLOW_RUN_ID_PATTERN.search()
        # (pas .match()) gère de toute façon aussi bien une valeur XCom
        # multi-lignes qu'une seule ligne.
        xcom_all=True,
    )

    check_drift_and_new_data() >> [train, skip]
    train >> promote(train.output)


retrain_on_drift()
