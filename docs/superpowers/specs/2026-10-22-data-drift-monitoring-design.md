# Monitoring de data drift (Evidently + MLflow) — Design

Date : 2026-10-22
Statut : validé (design), à décliner en plan d'implémentation
Branche : `romain`

## 1. Contexte et objectif

Le projet classifie des images de cellules sanguines. Pour détecter un drift
entre les images vues à l'entraînement et de nouvelles images (nouveau
scanner, nouvelle coloration, nouveaux cas cliniques), on veut produire un
rapport de data drift **sans jamais relire les images brutes
d'entraînement** : seule une référence compacte, calculée une fois à
l'entraînement et stockée comme artefact MLflow du run, doit suffire.

Les tests de drift usuels (Evidently `DataDriftPreset`, tests statistiques
par colonne) ont besoin d'échantillons des deux côtés de la comparaison —
impossible de les faire tourner sur un rapport déjà agrégé. Par ailleurs,
une image brute (pixels) n'est pas un objet sur lequel un test de drift a du
sens : il faut d'abord la réduire à des features numériques.

### Décisions structurantes (issues du brainstorming)

| Sujet | Décision |
|---|---|
| Nature de la référence | Deux DataFrames par run : descripteurs visuels bas-niveau + embeddings du modèle, sur un échantillon (pas les données brutes) |
| Échantillonnage | Stratifié par classe de la tâche, **200 images/classe** |
| Descripteurs visuels | Calculés sur transform minimal (resize + mise à l'échelle [0,1]), **sans** Reinhard/CLAHE/Normalize, pour ne pas masquer un drift de coloration/scanner |
| Embeddings | Features du modèle juste avant la tête de classification (`model.features(x)`), sur le transform d'évaluation complet |
| Portée modèles | **Tous** les modèles (`transfer`, `multi_archi`, `baseline`, `dataset`) doivent exposer `.features(x)` |
| Implémentation `.features(x)` pour les modèles "Sequential monolithique" | Refactor propre : `self.features` / `self.classifier` explicites (accepté : casse la compatibilité des checkpoints/objets MLflow pré-existants) |
| Robustesse | Toute étape embeddings (génération ou consommation) est **sautée silencieusement** (log, pas d'exception) si `.features()` est absent |
| Stockage de la référence | Fichiers dans `run_dir/drift_reference/`, repris automatiquement par `mlflow.log_artifacts(run_dir)` déjà existant — aucun nouvel appel MLflow côté entraînement |
| Métrique de drift — descripteurs | `Report(metrics=[DataDriftPreset(num_method="ks")])` — **pas** les défauts Evidently : au-delà de 1000 lignes de référence, `DataDriftPreset` bascule automatiquement (par colonne numérique) du test de Kolmogorov-Smirnov vers une distance de Wasserstein normée à seuil fixe 0.1, qui déclenche un faux positif quasi systématique sur un petit lot courant (100% de faux positifs mesurés à 20 images) — notre référence (200 images/classe) dépasse 1000 lignes dès 5-6 classes, donc `num_method="ks"` est forcé pour rester valide à toute taille de référence |
| Métrique de drift — embeddings | `evidently.metrics.EmbeddingsDrift(embeddings_name=..., drift_method=model(threshold=0.55))` (classifieur de domaine, `model()` importé de `evidently.legacy.metrics.data_drift.embedding_drift_methods`) — **ajoutée explicitement** à la liste de métriques du `Report`, pas incluse automatiquement par `DataDriftPreset` dans l'API native d'evidently≥0.7 (vérifié empiriquement : à la différence d'evidently 0.4.x, `DataDriftPreset` n'y câble pas `embeddings`/`embeddings_drift_method` dans son calcul) |
| Sortie du script de consommation | Fichier local unique uniquement (pas de run MLflow "monitoring" pour l'instant) |
| Résolution du modèle champion | Helper partagé déplacé de `server.py` vers `tracking.py`, étendu pour renvoyer aussi `run_id` |

### Hors périmètre (explicitement)

- Logger le rapport de monitoring dans un run MLflow dédié (pourra être
  ajouté plus tard derrière le même pattern `enabled()` que `tracking.py`).
- Endpoint bento pour déclencher le rapport depuis l'API de service (option
  b évoquée en amont ; CLI/batch seulement pour cette itération).
- Rétro-génération de la référence pour les runs déjà entraînés avant ce
  changement (cf. §6 Limitations).

## 2. Carte des modules cible

```
src/
  models/
    transfer.py            ~     + features(x) = self.backbone(x)  (déjà isolé)
    multi_archi_model.py   ~     __init__ refactorisé : self.features / self.classifier ; + features(x)
    baseline.py             ~     idem
    dataset_model.py        ~     idem
  features/
    drift_descriptors.py    NEW   compute_descriptors(images) -> pd.DataFrame ; transform minimal dédié
  experiments/
    drift_reference.py      NEW   sample_stratified, build_drift_reference -> DriftReference(descriptors, embeddings|None)
    runner.py                ~     hook : construit la DriftReference avant write_run_dir
    artifacts.py              ~     write_run_dir écrit drift_reference/{descriptors,embeddings}.parquet
    tracking.py                ~     accueille get_model_from_mlflow (déplacé depuis server.py), + run_id dans son retour
    server.py                  ~     importe get_model_from_mlflow depuis tracking.py au lieu de le définir
    drift.py                  NEW   run_drift_report(...) -> DriftReportResult ; logique de comparaison Evidently
    drift_cli.py               NEW   CLI argparse, pattern promote_cli.py
```

## 3. Flux de données

### 3.1 À l'entraînement (`run_experiment`, dans `runner.py`)

Entre le calcul de `result.preprocessing` et l'appel à `write_run_dir` :

1. `sample_df = sample_stratified(train_df, task.label_column, n_per_class=200, seed=config.seed)`.
2. `reference = build_drift_reference(sample_df, tb, best_model, task, device)` :
   - `descriptors = compute_descriptors(...)` : toujours calculé, sur les
     images de `sample_df` passées par un transform minimal (`ResizeWithPad`
     + mise à l'échelle [0,1] uniquement — pas de Reinhard/CLAHE/Normalize).
   - `embeddings = ...` : calculé via `best_model.features(x)` sur le même
     échantillon, mais passé par le transform d'évaluation complet
     (`tb.build(train_df, training=False)`) — cohérent avec ce que le modèle
     voit réellement en inférence. `None` si `not hasattr(best_model, "features")`.
3. `result.drift_reference = reference`.
4. `write_run_dir` (modifié) écrit `run_dir/drift_reference/descriptors.parquet`
   toujours, et `embeddings.parquet` si `reference.embeddings is not None`.
5. Aucun changement dans `tracking.log_run` : `mlflow.log_artifacts(str(run_dir))`
   (déjà appelé, `tracking.py:159`) embarque le nouveau sous-dossier
   automatiquement.

### 3.2 À la consommation (`drift_cli.py` → `drift.run_drift_report`)

1. Résolution du champion via le helper déplacé dans `tracking.py` :
   alias (défaut `"champion"`) → `model_version` → `run_id`, modèle PyTorch
   chargé, `config.yaml`, `preprocessing.json` (logique identique à
   l'actuel `get_model_from_mlflow`, étendue pour renvoyer `run_id`).
2. Téléchargement de la référence : `mlflow.artifacts.download_artifacts(
   f"runs:/{run_id}/drift_reference/descriptors.parquet")` (et
   `embeddings.parquet` si présent dans la liste des artefacts du run —
   vérifié avant de tenter le téléchargement, pas par try/except) puis
   `pd.read_parquet(...)`.
3. Recalcul des mêmes features sur les nouvelles images (`image_paths`
   fournis en argument) : mêmes fonctions `compute_descriptors` /
   `model.features(x)` que côté entraînement, avec le transform reconstruit
   via `rebuild_eval_transform`-like logic (déjà dans `inference.py`/`server.py`).
4. Fusion en **un seul DataFrame par côté** (référence / courant) :
   `combined = pd.concat([descriptors_df, embeddings_df], axis=1)` — valide
   car les deux DataFrames viennent du même échantillon, dans le même ordre
   de lignes (côté entraînement : `sample_df` ; côté consommation : les
   `image_paths` fournis). Si pas d'embeddings : `combined = descriptors_df`
   seul.
5. **Un seul** `Report`, construit une seule fois : les deux côtés sont
   enveloppés dans un `evidently.Dataset.from_pandas(df, data_definition=...)`,
   où `data_definition = DataDefinition(numerical_columns=[...colonnes
   descripteurs...], embeddings={"model_embedding": [...colonnes
   embeddings...]})` (le groupe `embeddings` est omis si la référence n'en a
   pas). `metrics = [DataDriftPreset(num_method="ks")]` (KS forcé pour toutes
   les colonnes numériques, pas les défauts Evidently — voir §1 : au-delà de
   1000 lignes de référence, le défaut bascule sur une distance de Wasserstein
   à seuil fixe qui est en faux positif quasi systématique sur un petit lot
   courant) ; si les embeddings sont présents, on ajoute explicitement
   `EmbeddingsDrift(embeddings_name="model_embedding", drift_method=model(threshold=0.55))`
   à la liste (voir §1 — l'API native d'evidently≥0.7 ne l'inclut plus
   automatiquement, contrairement à 0.4.x). Un seul appel
   `report.run(current_data=current_dataset, reference_data=reference_dataset)`
   (ordre des arguments : `current_data` en premier dans l'API native) produit
   un `Snapshot` unique, et `snapshot.save_html(...)` écrit un seul HTML avec
   les deux sections. Si les embeddings sont absents, le rapport ne contient
   naturellement que la section descripteurs — pas de cas particulier à coder.
   L'API native ne renvoyant pas de booléen "drift détecté" tout fait (contrairement
   à 0.4.x), les deux flags de `DriftReportResult` sont recalculés depuis la
   valeur mesurée et le seuil/`drift_share`, relus directement depuis le
   `config` de la métrique correspondante dans `snapshot.dict()` (source
   unique, pas de constante dupliquée).
6. Écriture locale uniquement : `drift_report.html` (sortie de
   `snapshot.save_html(...)`) et `drift_summary.json` avec les scores agrégés
   (drift détecté oui/non par section, score numérique si disponible, via
   `snapshot.dict()`).

## 4. Interfaces

```python
# src/models/* (toutes les classes)
def features(self, x: torch.Tensor) -> torch.Tensor: ...

# src/features/drift_descriptors.py
def compute_descriptors(image_paths_or_tensors) -> pd.DataFrame: ...

# src/experiments/drift_reference.py
@dataclass
class DriftReference:
    descriptors: pd.DataFrame
    embeddings: pd.DataFrame | None

def sample_stratified(df, label_column, n_per_class, seed) -> pd.DataFrame: ...
def build_drift_reference(sample_df, transform_builder, model, task, device) -> DriftReference: ...

# src/experiments/tracking.py (déplacé + étendu)
def get_model_from_mlflow(model_name, client, alias="champion"):
    ...
    return transform, model, task, run_id  # run_id en plus de l'existant

# src/experiments/drift.py
@dataclass
class DriftReportResult:
    descriptors_drifted: bool
    embeddings_drifted: bool | None  # None si section absente
    report_path: Path

def run_drift_report(
    model_name: str,
    image_paths: Sequence[str | Path],
    image_files: Sequence[bytes] | None = None,
    alias: str = "champion",
    output: Path = Path("drift_report"),
) -> DriftReportResult: ...
```

**Convention de colonnes** (nécessaire pour le groupe `embeddings` du
`DataDefinition` du §3.2) : `build_drift_reference` nomme les colonnes
d'embedding `emb_0, emb_1, ..., emb_{n-1}` (une colonne par dimension du
vecteur de features) — cette liste de noms est le groupe passé à
`DataDefinition(embeddings={"model_embedding": [...]})` côté `drift.py`.
`compute_descriptors` évite ce préfixe pour ne jamais collisionner.

## 5. Gestion des erreurs

- `.features()` absent sur le modèle (entraînement ou consommation) :
  comportement normal, pas une erreur — section embeddings simplement
  absente de la référence / du rapport, avec un message informatif.
- Référence `descriptors.parquet` absente du run champion (run plus ancien
  que cette fonctionnalité, ou run dont `output.save_checkpoint=False`) :
  erreur explicite et bloquante à la résolution du champion — contrairement
  au cas embeddings, il n'y a rien à rapporter sans la référence
  descripteurs ; le script doit s'arrêter avec un message clair plutôt que
  produire un rapport vide.
- Échec MLflow (serveur injoignable) lors de la résolution du champion :
  propagé tel quel (pas de no-op comme `tracking.log_run` — ici, contrairement
  à l'entraînement, le monitoring *est* l'action demandée ; le faire échouer
  silencieusement serait trompeur).

## 6. Limitations connues (à documenter, pas à résoudre ici)

- **Migration des champions existants — casse plus grave qu'une simple perte
  de drift-embeddings** : le refactor `self.features`/`self.classifier`
  renomme un attribut d'instance (`self.model` → `self.features`/
  `self.classifier`). `mlflow.pytorch.load_model` dépickle l'objet module en
  conservant ses **anciens** attributs d'instance (un seul `self.model`,
  `nn.Sequential` monolithique), alors que `forward()` est résolu depuis la
  **classe courante** au moment de l'appel (jamais gelé dans le pickle) et
  fait désormais `self.classifier(self.features(x))` — des attributs que ce
  vieux pickle n'a jamais eus. Concrètement, charger puis faire une inférence
  avec un champion `multi_archi`/`baseline`/`dataset` enregistré avant ce
  changement **ne se limite pas** à perdre le support drift-embeddings : ça
  plante immédiatement sur **tout** appel `forward()`, pas seulement sur le
  chemin drift — `AttributeError` pour `baseline`/`dataset`
  (`self.features` inexistant sur le vieux pickle), `TypeError: 'list' object
  is not callable` pour `multi_archi` (son ancien `self.classifier` était la
  liste Python passée au constructeur, pas un `nn.Module`). `TransferModel`
  n'est pas concerné : `.features()` y est une **méthode** ajoutée à la
  classe (résolue dynamiquement, pas un attribut d'instance pické), donc un
  vieux pickle `TransferModel` continue de fonctionner sans rien changer. Pour
  éviter un crash opaque, `tracking.get_model_from_mlflow` lève désormais un
  `ValueError` explicite dès que le modèle chargé n'a pas `.features`
  (`hasattr(model, "features")`), avant même la première inférence — il
  faudra ré-entraîner ou re-logger ces champions pour les rendre à nouveau
  utilisables. Le drift-descripteurs, lui, n'est jamais affecté par ce
  refactor (il ne dépend d'aucun attribut de modèle).
- **Cohérence référence ↔ modèle** : la référence embeddings n'a de sens que
  comparée à des embeddings produits par le **même** modèle (même espace
  latent). Le design l'assure *structurellement* : la référence et le modèle
  sont toujours chargés depuis le **même** `run_id` (celui du champion
  résolu), jamais mélangés entre runs — pas de tag de version supplémentaire
  nécessaire.
- Pas de suivi temporel du drift (chaque exécution du script est indépendante)
  tant que le logging MLflow côté monitoring reste hors périmètre.

## 7. Tests

- `compute_descriptors` : valeurs attendues sur une image synthétique
  (couleur unie connue → luminosité/histogramme prévisibles).
- `sample_stratified` : couverture de toutes les classes présentes,
  déterminisme avec un `seed` fixé, comportement si une classe a moins de
  200 images.
- `.features(x)` sur chacune des 4 classes de modèle : forme de sortie
  cohérente (vecteur `(B, num_features)`), absence de la tête de
  classification dans le calcul.
- `build_drift_reference` : `embeddings is None` quand le modèle n'a pas
  `.features`.
- `run_drift_report` (intégration légère, run MLflow simulé/fixture) :
  rapport produit avec les deux sections quand tout est présent ; section
  embeddings absente avec message si `embeddings.parquet` manque ; erreur
  explicite si `descriptors.parquet` manque.

## 8. Nouvelle dépendance

`evidently` ajoutée à `pyproject.toml`, épinglée (`evidently>=0.7.23,<0.8`).

Historique : la première implémentation (2026-10-22) avait épinglé
`evidently>=0.4.36,<0.5` — seule version résolue à l'époque à cause du
conflit `evidently<0.7` exige `plotly<6` alors que ce projet imposait
`plotly>=6`. Une fois cette contrainte retirée (migration du graphique PCA
3D de plotly vers matplotlib/seaborn), le projet est passé à l'API native
`evidently>=0.7` (`Dataset`/`DataDefinition`/`Report`/`presets.DataDriftPreset`
+ `metrics.EmbeddingsDrift`), cf. §1/§3.2/§4 — le code n'utilise plus l'API
legacy `ColumnMapping`/`evidently.report.Report`/`evidently.metric_preset.*`
sauf pour l'import ponctuel de `evidently.legacy.metrics.data_drift.embedding_drift_methods.model`
(la méthode de drift des embeddings, inchangée entre les deux générations
d'API).
