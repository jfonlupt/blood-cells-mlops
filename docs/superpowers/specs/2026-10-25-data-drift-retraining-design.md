# Réentraînement déclenché par data drift (Airflow + DVC/git + MLflow) — Design

Date : 2026-10-25
Statut : design, à décliner en plan d'implémentation
Branche : `romain`

## 1. Contexte et objectif

But pédagogique/démonstratif : simuler un data drift en production et montrer
la chaîne complète qui y réagit. Scénario :

1. Un modèle est entraîné **uniquement sur munich**, déployé via
   `blood_cells_server` (BentoML).
2. Un script simulateur envoie des requêtes `/evaluate` dont le mélange glisse
   progressivement de munich vers barcelona — simulant l'arrivée d'une
   nouvelle population de patients/scanner en production.
3. Chaque image barcelona vue est persistée (DVC + git) au fur et à mesure,
   comme le ferait un vrai pipeline de collecte de données de production.
4. Une fois la dérive confirmée de façon soutenue, un réentraînement se
   déclenche automatiquement sur munich + les données barcelona accumulées.
5. Le nouveau modèle n'est promu "champion" que s'il ne régresse pas sur la
   distribution d'origine et qu'il gère mieux la distribution actuelle que
   l'ancien champion.
6. Tout l'enchaînement détection → collecte → réentraînement → promotion est
   orchestré par Airflow, déjà présent dans `docker-compose.yaml` mais sans
   DAG pour l'instant (`dags/` vide).

Ce document ne couvre que l'orchestration de bout en bout. Le calcul du
drift lui-même (Evidently, références `descriptors`/`embeddings` loggées en
artefact MLflow par run, endpoint `/evaluate`) est déjà conçu et implémenté —
voir [`2026-10-22-data-drift-monitoring-design.md`](2026-10-22-data-drift-monitoring-design.md).
Ce document-ci part de cet existant sans le redécrire.

### Ce qui existe déjà (vérifié dans le code, pas à refaire)

- `src/experiments/drift.py` : calcule `descriptors_drifted`/`embeddings_drifted`
  (booléens) contre la référence du champion MLflow.
- `src/experiments/server.py` (`/evaluate`) : calcule déjà le rapport de
  drift Evidently **et** les métriques de qualité (accuracy/precision/recall/F1)
  à partir des labels fournis, et les expose en gauges Prometheus
  (`model_descriptor_columns_drifted_share`, `model_embeddings_drift`,
  `model_f1_score`, etc.).
- `deployment/grafana/provisioning/alerting/rules.yml` : règles d'alerte déjà
  provisionnées sur ces mêmes gauges (F1 < 0.7, drift descripteurs > 0.6,
  drift embeddings > 0.55), pensées pour notifier des humains.
- `src/experiments/tracking.py` : chaque run MLflow est déjà tagué
  `mlflow.source.git.commit` (via `GIT_COMMIT`, alimenté par
  `git rev-parse HEAD` dans le `Makefile` au build de l'image
  `blood_cells_training`).
- `src/helpers/data_drift.py` : brouillon du générateur de batches
  munich→barcelona progressifs (`generate_data_drift`) et d'un script
  client qui appelle `/predict` puis `/evaluate` ; le bloc de persistance
  DVC/git est stubé en commentaire.
- `configs/densenet_data_drift.yaml` : config de réentraînement
  `datasets: [munich, data_drift]`.
- `src/config.py`/`src/constants.py` : clé de dataset `data_drift`
  (`DRIFT_KEY`) déjà ajoutée, mappée sur `BARCELONA_TO_FINAL`.
- `.dvc/config` : `cache.type = "hardlink,symlink"`, remote `s3://blood-cells-.../dvc`.
- `src/experiments/tracking.py` (`register_model`) : **déjà** exécuté sur
  chaque run (`save_checkpoint=True`), pose automatiquement un alias
  `"challenger"` si `test_f1_macro` (métrique et marge +0.01 codées en dur)
  dépasse le champion — mécanisme générique, indépendant de ce design, à ne
  pas casser pour les runs hors drift.
- `src/experiments/promote.py`/`promote_cli.py` : outil **sans rapport**
  avec la promotion MLflow (malgré le nom) — sélectionne les meilleurs runs
  du leaderboard et prolonge leur entraînement de N epochs. Ne pas confondre
  avec la promotion champion/challenger de ce design.
- `src/features/dataframe.py` (`get_images`/`create_dataframe`) : scan
  **récursif** (`glob("**/*.jpg")`), label dérivé du **nom de fichier**
  (`p.stem.split("_")[0]`), indépendant de la profondeur du dossier — point
  important, réutilisé par plusieurs décisions ci-dessous.
- `docker-compose.yaml` : Airflow 3.3.2 (CeleryExecutor + postgres + redis),
  `dags/` monté mais vide.

## 2. Décisions structurantes (issues du brainstorming)

| Sujet | Décision | Pourquoi |
|---|---|---|
| Déclenchement du réentraînement | **Airflow interroge Prometheus** (pull), pas de webhook Grafana→Airflow | Investigation : Airflow 3.3.2 utilise par défaut le "simple auth manager" (JWT via `POST /auth/token`), incompatible avec un unique appel webhook sortant de Grafana sans relais supplémentaire (FAB auth manager ou token statique, deux options avec leurs propres coûts). Faire d'Airflow l'appelant (sortant) évite tout ce problème : Airflow n'a jamais besoin d'exposer son API à qui que ce soit. |
| Rôle de Grafana | Conservé **uniquement** pour notifier des humains (règles déjà en place), découplé de l'automatisation | Deux responsabilités séparées, ni l'une ni l'autre ne dépend de l'autre. |
| Unification des seuils Evidently/Grafana | Exposer un gauge booléen calculé serveur, ex. `model_drift_detected{model=...}` = 1 si `descriptors_drifted` ou `embeddings_drifted` (déjà calculés dans `drift.py`), 0 sinon | Évite de dupliquer les seuils numériques (0.6, 0.55...) à la fois dans le code Evidently et dans le YAML Grafana. Python/Evidently reste la seule source de vérité ; les règles Grafana et le polling Airflow deviennent de simples vérifications `> 0`. |
| Fenêtre de détection | Le polling Airflow lit une **plage** (`query_range`), pas l'instant courant, et exige qu'une part suffisante des points soit à 1 | Évite de réagir à un seul batch limite pendant que le mélange munich/barcelona est encore proche de 50/50. |
| Garde-fou anti-réentraînement redondant | Comparaison par **commit git**, pas de compteur/Variable Airflow séparée | Chaque run d'entraînement tague déjà `mlflow.source.git.commit`. La tâche de vérification calcule le dernier commit touchant le chemin des données drift (`git log -1 -- <chemin>`, après `git fetch`), va chercher le run MLflow le plus récent (peu importe s'il a été promu), et teste `git merge-base --is-ancestor <commit_données> <commit_du_run>` : si vrai, ces données étaient déjà incluses → skip ; sinon → il y a du nouveau → on continue. MLflow reste la seule source de vérité, aucun état à maintenir côté Airflow. |
| Granularité des commits de collecte | Le simulateur (Phase I) **batch** ses commits (pas un commit par appel API) | Sert de seuil minimal implicite : un nouveau commit doit représenter un lot d'images significatif, pas 1-2 images. |
| Concurrence | `max_active_runs=1` sur le DAG | Empêche deux réentraînements simultanés ; combiné au garde-fou par commit git ci-dessus pour empêcher aussi deux réentraînements **consécutifs** sur les mêmes données. |
| Jeu(x) d'évaluation pour la promotion | **Deux** jeux séparés, pas un seul jeu "gold" fixé une fois | Un jeu figé une fois ne peut jamais inclure les nouvelles images (sinon il ne serait pas comparable dans le temps) ; mais un jeu qui doit rester à jour ne peut pas aussi servir de référence stable. Un seul jeu ne peut pas remplir les deux rôles. |
| → Socle fixe | Réserve de **munich uniquement**, prélevée une fois, avant le tout premier entraînement, exclue pour toujours de tout entraînement (initial et tous les réentraînements ultérieurs) | Détecte une régression/oubli sur la distribution d'origine du modèle. N'a pas besoin de grandir : la distribution de base ne dérive pas. |
| → Jeu récent | Fraction (ex. 80/20) de **chaque nouvelle image barcelona** réservée **au moment de la capture** (Phase I), jamais utilisée à l'entraînement, qui grandit en continu | Garanti non-vu à l'entraînement par construction (réservé dès l'arrivée, pas recalculé après coup) ; reflète la distribution actuelle, contrairement au socle fixe. |
| Implémentation physique des deux jeux | **Dossiers physiques** `data/train/<dataset>/...` et `data/test/<dataset>/...`, hardlinks par-dessus les dossiers plats existants (pas de migration, pas de manifeste/colonne de rôle) | `get_images`/`create_dataframe` scannent déjà récursivement et dérivent le label du nom de fichier — zéro changement de logique de lecture, juste de nouvelles clés `DATASET_PATHS`. `.dvc/config` a déjà `cache.type = "hardlink,symlink"` : pas d'espace disque supplémentaire, ni localement (hardlink manuel) ni pour quiconque re-clone (dédup par hash de contenu dans le cache DVC). Les dossiers plats existants (`PKG-AML-Cytomorphology_LMU/`, `PBC_dataset_normal_DIB/`) et leurs `.dvc` restent intacts — cette nouvelle vue est additive. |
| Relation avec le split train/test habituel | Ce découpage physique est un palier **au-dessus** de `test_size`/`split_off_test` | `split_off_test` continue de re-découper, à chaque run, ce qu'il y a dans le dossier `train/` physique — il ne disparaît pas, il opère juste sur un périmètre déjà expurgé des deux jeux d'évaluation ci-dessus. |
| Déclenchement de l'entraînement | `DockerOperator` (ou équivalent) depuis Airflow, sur l'image `blood_cells_training` existante, avec `configs/densenet_data_drift.yaml` | Pattern Airflow standard pour orchestrer un job déjà packagé en image ; pas de provider Airflow dédié à Prometheus (vérifié), donc la tâche de polling est un appel HTTP simple enveloppé dans un `PythonOperator`/branche — idiomatique aussi. |
| Fraîcheur de `mlflow.source.git.commit` sur le run déclenché | **Override runtime de l'`ENV GIT_COMMIT`** au lancement du container (`DockerOperator(environment={"GIT_COMMIT": <HEAD actuel>})`), pas de rebuild de l'image `blood_cells_training` à chaque cycle | Trouvé en relecture : `GIT_COMMIT` n'est aujourd'hui fixé qu'au **build** de l'image (`docker/Dockerfile` ARG/ENV, déclenché par `make build-pipeline`, séparé de `make pipeline`). Sans ce correctif, tous les runs déclenchés par Airflow tagueraient le même commit figé au dernier build, et le garde-fou anti-réentraînement redondant (comparaison par commit git, cf. ligne ci-dessus) deviendrait inopérant — il verrait presque toujours "nouvelles données" puisque le commit comparé ne bougerait jamais. Docker autorise de surcharger un `ENV` de l'image au lancement, sans rebuild : la tâche `check_drift_and_new_data` calcule déjà le HEAD courant pour sa propre vérification, il suffit de le transmettre à la tâche suivante. |
| Portée de la clé `munich` | **Nouvelle clé** dédiée (ex. `munich_train`), `DATASET_PATHS[MUNICH_KEY]` **inchangée** | `munich` est référencée par ~12 configs et ~14 notebooks sans rapport avec cette fonctionnalité (dont les travaux sur la matrice de coût de criticité). Réduire `DATASET_PATHS[MUNICH_KEY]` globalement casserait silencieusement la reproductibilité de tout ce qui l'utilise déjà. Le socle fixe (§ ligne "Socle fixe") doit donc vivre derrière une clé séparée, utilisée uniquement par `configs/densenet_data_drift.yaml` et ses successeurs. |
| Résolution du challenger en Phase IV | Le challenger est le run dont Airflow vient de déclencher l'entraînement — **résolu par `run_id`** (XCom depuis la tâche d'entraînement, ou à défaut requête MLflow sur le tag `mlflow.source.git.commit` fixé juste au-dessus), **pas** via l'alias `"challenger"` existant | `register_model` pose déjà un alias `"challenger"` automatiquement sur **tout** run, avec sa propre métrique (`test_f1_macro`, marge +0.01 codée en dur) et son propre split de test (le holdout classique, pas le socle/jeu récent de ce design) — mécanisme générique à ne pas désactiver (il sert aussi hors drift). S'appuyer sur cet alias pour Phase IV serait fragile : rien ne garantit qu'il soit posé sur CE run précis (son propre critère peut décider autrement), et l'évaluation de Phase IV doit se faire sur les jeux dédiés (socle + jeu récent), pas sur le split classique. Les deux mécanismes coexistent sans interagir : l'alias `"challenger"` de `register_model` peut continuer à exister, Phase IV ne le lit jamais. Nécessite que l'entrypoint d'entraînement expose son `run_id` (print/fichier) pour que la tâche suivante le récupère. |
| Métrique de décision de promotion | **F1** pour l'instant, mais l'implémentation doit rendre ce choix **facilement remplaçable dans le code** (ex. constante/registre nommé plutôt qu'un calcul en dur) | Exigence explicite : le projet a par ailleurs une loss pondérée par matrice de coût de criticité jugée plus pertinente (cf. mémoire du projet) ; on ne tranche pas ici lequel des deux est définitivement le bon, juste que le changement doit être trivial plus tard. |
| Garde-fou de promotion | Le challenger doit : ne pas régresser significativement sur le socle fixe, ET être au moins aussi bon que le champion sur le jeu récent | Protège contre l'oubli catastrophique tout en s'assurant que le réentraînement a bien amélioré la gestion de la dérive — le point même de déclencher un réentraînement. |
| Rafraîchissement du cache serveur | **TTL configurable** sur `BloodCellModelService._get_champion`, pas d'invalidation événementielle ni de redémarrage du container | Cohérent avec la logique "pull" déjà adoptée pour la détection : le serveur se revérifie périodiquement tout seul, sans canal de coordination Airflow→serveur à construire. |
| Credentials git/dvc pour la Phase I (push) | **Exécution manuelle**, le simulateur réutilise les credentials déjà configurés sur le poste de l'opérateur (clé SSH git, profil AWS `blood-cells` déjà référencé dans `.dvc/config.local`) — pas de nouveau service `docker-compose`, pas de credential d'écriture dédiée provisionnée | Pour ce projet pédagogique, provisionner une credential d'écriture séparée (clé de déploiement git, utilisateur IAM scopé) sur un repo et un bucket S3 partagés serait un coût/risque disproportionné par rapport au but de démonstration. Option alternative (service dédié + credentials scopées) explicitement écartée pour cette itération, pas définitivement. |

## 3. Les 5 phases

```
Phase I   (simulateur, hors Airflow)
  /evaluate (munich→barcelona progressif)
    → pour chaque image barcelona du batch :
        - tirage train_pool (80%) / eval_holdout (20%)
        - hardlink dans data/train/data_drift/... ou data/test/data_drift/...
        - dvc add, commit git (batché), dvc push, git push

Phase II  (Airflow, DAG planifié, ex. toutes les 5-10 min)
  check_drift_and_new_data :
    - query_range Prometheus sur model_drift_detected → dérive soutenue ?
    - dernier commit touchant data/train/data_drift.dvc vs mlflow.source.git.commit
      du run le plus récent (ancestor check) → nouvelles données ?
    - les deux vrais → continue ; sinon → skip (branche no-op)

Phase III (Airflow, DockerOperator)
  trigger_training :
    - lance blood_cells_training avec configs/densenet_data_drift.yaml
    - override GIT_COMMIT=<HEAD courant> au lancement (pas de rebuild)
    - lit data/train/munich + data/train/data_drift (jamais les dossiers test/)
    - récupère le run_id MLflow du run lancé (XCom)

Phase IV  (Airflow, PythonOperator + API MLflow)
  evaluate_and_promote :
    - résout le challenger par run_id (pas par l'alias "challenger" existant,
      posé par un mécanisme générique indépendant avec son propre critère)
    - évalue champion ET challenger sur data/test/munich (socle) et
      data/test/data_drift (jeu récent)
    - si challenger ok sur les deux critères → set_registered_model_alias(champion)
    - sinon → rien ne change, run tracé pour audit

Phase V   (blood_cells_server, continu, indépendant d'Airflow)
  _get_champion recharge depuis MLflow si l'entrée en cache a plus de
  TTL_CHAMPION_CACHE (config) minutes
```

## 4. Hors périmètre (explicitement)

- Le calcul du drift Evidently lui-même (cf. doc du 2026-10-22).
- Rétro-compatibilité avec un historique de champions antérieurs à ce
  design — on part du principe qu'un premier champion munich-only est
  entraîné après la mise en place du socle fixe (cf. §5, ordre d'opérations).
- Permettre de ré-ouvrir plus tard la question push/Grafana : tranchée pour
  ce projet, pas un choix réversible sans re-brainstormer.
- Alertmanager Prometheus natif (le projet utilise l'alerting intégré de
  Grafana directement sur le datasource Prometheus, pas un Alertmanager
  séparé) — non remis en cause ici.

## 5. Ce qui reste à l'implémenteur

- **Ordre d'opérations au tout premier déploiement** : le socle fixe munich
  doit être prélevé et figé **avant** le premier entraînement du premier
  champion, sans quoi le premier champion aurait pu voir ces images.
- **Noms de constantes/chemins** (proposition, pas figée) :
  `data/train/munich`, `data/train/data_drift`, `data/test/munich`,
  `data/test/data_drift` ; clés `DATASET_PATHS` type `munich_train`,
  `munich_eval`, `data_drift_train`, `data_drift_eval` (`MUNICH_KEY` existant
  inchangé, cf. §2 "Portée de la clé munich").
- **Exposer le `run_id` MLflow** depuis l'entrypoint d'entraînement
  (`src/experiments/run.py` ou équivalent) pour que la tâche Airflow
  suivante puisse le récupérer (XCom via stdout, ou fichier dans
  `results/runs/<hash>/`) — nécessaire pour la résolution du challenger en
  Phase IV (cf. §2).
- **Ratio exact** de réservation train_pool/eval_holdout à la capture
  (80/20 illustratif) et seuil de nombre minimal d'images avant de
  considérer le jeu récent "assez grand" pour comparer dessus.
- **Stratification** du tirage train_pool/eval_holdout par classe une fois
  qu'il y a assez de volume (au tout début, un tirage par image risque
  d'affamer les classes rares d'un côté ou l'autre).
- **Mécanisme concret de métrique switchable** (Phase IV) : forme exacte du
  point d'extension (ex. petit registre `{"f1": ..., "criticality_loss": ...}`
  + une constante qui sélectionne laquelle) — le besoin de switch facile est
  une contrainte dure, la forme du code ne l'est pas.
- **Marge de tolérance** de la comparaison champion/challenger (égalité
  stricte, ou petite marge epsilon pour ne pas promouvoir sur du bruit).
- **Intervalle de polling** du DAG (`schedule_interval`) et fenêtre de
  `query_range` pour juger une dérive "soutenue".
- **Accès infra pour Airflow** : accès réseau à Prometheus (déjà sur le même
  réseau docker-compose), un checkout git avec `fetch` en lecture (pas de
  push nécessaire depuis Airflow — le push vient de la Phase I, côté
  simulateur), et accès au socket docker + aux mêmes volumes que
  `blood_cells_training` pour le `DockerOperator`.
- **Credentials de push git/dvc pour le simulateur (Phase I)** : où ils
  vivent et comment ils sont scopés — distinct des accès Airflow (lecture
  seule) ci-dessus.
- **Valeur du TTL** de cache côté serveur (Phase V) et son nom de variable
  de configuration.

## 6. Risques connus / limitations

- La latence de détection est bornée par l'intervalle de polling (minutes),
  pas quasi-instantanée comme l'aurait été un webhook — jugé acceptable
  pour ce projet.
- Les dossiers `data/train/` et `data/test/` dupliquent la structure DVC
  existante (deux arborescences trackées pour le même contenu) : pas de
  surcoût d'espace disque, mais un historique git plus volumineux au moment
  de la réorganisation initiale (chaque fichier "apparaît" à son nouveau
  chemin), une seule fois.
- `os.link()` exige que les deux chemins soient sur le même système de
  fichiers — vrai aujourd'hui (tout sous `DATA_DIR`), à surveiller si un
  jour une partie des données est montée sur un volume Docker séparé.
- Le choix F1 vs. loss pondérée par criticité pour la promotion n'est pas
  tranché définitivement par ce document — seule la facilité de bascule
  est une exigence.
