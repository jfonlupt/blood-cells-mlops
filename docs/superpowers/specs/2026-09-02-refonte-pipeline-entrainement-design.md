# Refonte du pipeline d'entraînement — Design

Date : 2026-09-02
Statut : validé (design), à décliner en plan d'implémentation
Branche : `romain_v2`

## 1. Contexte et objectif

`src/train.py` est aujourd'hui un `main()` argparse (~20 flags) qui appelle un
`run()` monolithique de ~150 lignes. Ce `run()` fait tout : construction du
DataFrame, choix du modèle / de la loss / des métriques selon une string `mode`,
split, calcul des stats Reinhard, closure `transform_fn`, instanciation des
datasets, résolution du device, factories optimizer / scheduler, appel à
`model.fit(...)`, tracé des figures, évaluation test.

En parallèle, `BaseModel.fit()` (`src/models/base_model.py`) porte toute
l'orchestration de l'entraînement (split train/val, validation croisée,
construction des transforms par fold, scheduler, early stopping) **mélangée**
au `nn.Module`.

### Objectifs

1. Séparer les responsabilités en unités testables indépendamment.
2. Pouvoir lancer un **grid search** sur les hyperparamètres de : transformations /
   augmentations, loss, scheduler, optimizer, entraînement.
3. Passer proprement d'un **type de problème** à un autre : classification du
   dataset d'origine, des types cellulaires, de la criticité (3 classes) ou de
   la criticité binaire.
4. Conserver **toutes** les sorties actuelles (courbes loss/metrics/lr par epoch,
   figures et CSV de métriques test, matrice de confusion) et les ranger par run.

### Décisions structurantes (issues du brainstorming)

| Sujet | Décision |
|---|---|
| Pilotage des runs | Fichiers YAML → dataclasses typées, + runner séquentiel |
| Loader / validation de config | **pydantic v2** (ajout de dépendance) |
| Grid search | `sklearn.model_selection.ParameterGrid` (déjà dépendance), produit cartésien, exécution séquentielle locale |
| Portée d'une invocation grid | **une invocation = une tâche** (`--task`) |
| Boucle d'entraînement | **Extraire un `Trainer`** ; les modèles redeviennent de purs `nn.Module` |
| Style CLI | `--config` + overrides ponctuels `--set a.b=val` |
| Artefacts par run | dossier par run + `leaderboard.csv` agrégé + checkpoint du meilleur modèle |
| `WeightedRandomSampler` (todo #3) | **hors périmètre**, seam laissé propre |
| Assemblage des objets depuis la config | fonctions `build_*` libres à côté de leur domaine (« registry-lite »), `config.py` sans import torch |

## 2. Carte des modules cible

```
src/
  config.py            NEW   arbre pydantic ExperimentConfig + load_config + overrides --set
  tasks.py             NEW   TaskSpec + registre TASKS (enrobe les mappings de constants.py)
  constants.py         inchangé
  features/
    dataframe.py       NEW   create_dataframe, make_update_data      (extrait de dataset.py)
    dataset.py         SLIM  BloodCellDataset (une seule classe), DatasetTransform
    stats.py           NEW   compute_dataset_mean_std + moyennage stats Reinhard (extrait de transform.py)
    transform.py       ~     build_transform(TransformConfig, …) + classes ResizeWithPad/Reinhard/CLAHE
  models/
    baseline.py        REN   ex train_model.py — nn.Module pur
    dataset_model.py   ~     nn.Module pur (n'hérite plus de BaseModel)
    __init__.py        NEW   build_model(ModelConfig, task)
    losses.py          ~     + build_loss(LossConfig, task, train_df)
    metrics.py         ~     build_metrics(task, for_test=False) ; display_metrics inchangé
  training/
    trainer.py         NEW   Trainer — boucle fit/evaluate/predict (ex base_model.py)
    callbacks.py       MOVE  ex models/callbacks.py (API EarlyStopping inchangée)
    components.py      NEW   build_optimizer, build_scheduler, TransformBuilder, make_splits, resolve_device
  experiments/
    runner.py          NEW   run_experiment(cfg) -> RunResult
    grid.py            NEW   expand_grid, run_grid, append leaderboard
    artifacts.py       NEW   layout du dossier de run, save/load, leaderboard.csv
    run.py             NEW   CLI run unique : python -m src.experiments.run
    grid_cli.py        NEW   CLI grid       : python -m src.experiments.grid
  train.py             SUPPR remplacé par experiments/run.py

configs/
  base.yaml            NEW   valeurs par défaut de ExperimentConfig, explicites
  grids/
    preprocessing.yaml NEW   reinhard / clahe / jpeg on-off
    augmentation.yaml  NEW   p_rotation, p_brightness, p_contrast, *_delta
    imbalance.yaml     NEW   loss.name, loss.beta, loss.gamma

docs/superpowers/specs/2026-09-02-refonte-pipeline-entrainement-design.md   (ce fichier)
```

`src/models/base_model.py` est **supprimé**. `BaselineClassifier` et
`DatasetClassifier` redeviennent de purs `nn.Module`. Les sous-classes
`DomainDataset` / `CriticalityDataset` sont **supprimées** — leur seule
différence (dtype / forme du label) devient un attribut de `TaskSpec`.

## 3. Schéma de configuration (pydantic v2)

`src/config.py` — **aucun import torch**, uniquement données + validation.
Tous les modèles ont `model_config = ConfigDict(extra="forbid")` : une clé mal
orthographiée dans un YAML lève une `ValidationError`.

```python
class DataConfig(BaseModel):
    datasets: list[Literal["barcelona", "munich"]] = ["barcelona", "munich"]
    n_images: int = 0                     # 0 = tout
    test_size: float = 0.15
    stratify: Literal[
        "no", "type", "dataset", "type_dataset", "criticality", "binary_criticality"
    ] = "type_dataset"

class TransformConfig(BaseModel):
    img_size: tuple[int, int] = (224, 224)
    reinhard: bool = True
    reinhard_samples: int = 500
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    jpeg: bool = True
    jpeg_quality: int = 35
    # augmentation (train uniquement)
    p_rotation: float = 0.9
    p_brightness: float = 0.5
    p_contrast: float = 0.5
    brightness_delta: float = 0.15
    contrast_delta: float = 0.15

class LossConfig(BaseModel):
    name: Literal["cross_entropy", "focal", "cost_sensitive"] = "cross_entropy"
    beta: float = Field(0.9999, gt=0, lt=1)     # focal / cost_sensitive
    gamma: float = 2.0                           # focal
    extra_weight: dict[str, float] = {}          # focal — boost clinique manuel
    lambda_cost: float = 1.0                     # cost_sensitive

class OptimConfig(BaseModel):
    name: Literal["adam", "sgd"] = "adam"
    lr: float = 1e-2
    weight_decay: float = 0.0

class SchedulerConfig(BaseModel):
    name: Literal["none", "plateau", "cosine", "step"] = "none"
    patience: int = 3          # plateau
    lr_min: float = 0.0
    step_gamma: float = 0.1    # step

class ModelConfig(BaseModel):
    name: Literal["baseline", "transfer"] = "baseline"

class TrainConfig(BaseModel):
    nb_epoch: int = 3
    batch_size: int = 32
    validation: Literal["none", "holdout", "cross"] = "holdout"
    val_size: float = 0.15
    cross_val_folds: int = 3
    early_stopping_patience: int = 0            # 0 = désactivé
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

class OutputConfig(BaseModel):
    root: Path = Path("results")
    save_checkpoint: bool = True
    save_figures: bool = True

class ExperimentConfig(BaseModel):
    task: Literal["dataset", "type", "criticality", "binary_criticality"] = "type"
    seed: int = 0
    data: DataConfig = DataConfig()
    transform: TransformConfig = TransformConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    optim: OptimConfig = OptimConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    train: TrainConfig = TrainConfig()
    output: OutputConfig = OutputConfig()

    @model_validator(mode="after")
    def _check_validation_requirements(self):
        needs_val = (
            self.train.early_stopping_patience > 0
            or self.scheduler.name == "plateau"
        )
        if needs_val and self.train.validation == "none":
            raise ValueError(
                "scheduler=plateau et early_stopping_patience>0 exigent "
                "validation='holdout' ou 'cross'."
            )
        return self
```

### Chargement et overrides

```python
def load_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    merged = _deep_merge(raw, _parse_overrides(overrides or []))
    return ExperimentConfig.model_validate(merged)
```

- `_parse_overrides(["train.lr=1e-3", "loss.name=focal"])` → dict imbriqué
  `{"train": {"lr": 1e-3}, "loss": {"name": "focal"}}`. Chaque valeur passe par
  `ast.literal_eval` (donc `1e-3`, `true`, `[1, 2]`, `(224, 224)` sont typés) ;
  en cas d'échec, la valeur reste une `str`.
- `_deep_merge(base, patch)` : fusion récursive dict par dict, `patch` prioritaire.
- La re-validation pydantic après merge attrape toute combinaison invalide.

## 4. Abstraction de tâche

`src/tasks.py` centralise tout ce qui est aujourd'hui dispersé en
`if mode == "..."` (dans `train.py`, `features/dataset.py:make_update_data`,
`models/metrics.py`).

```python
@dataclass(frozen=True)
class TaskSpec:
    name: str
    label_column: str                 # colonne du DataFrame servant de label
    class_names: list[str]
    num_classes: int
    kind: Literal["binary", "multiclass"]
    label_dtype: torch.dtype          # long partout ; float32 pour "dataset" (BCE)
    target_shape: Literal["scalar", "vector1"]  # "vector1" -> label de forme (1,)
    default_loss: Literal["cross_entropy", "bce"]

def get_task(name: str) -> TaskSpec: ...

TASKS: dict[str, TaskSpec] = {
    "dataset": TaskSpec(
        name="dataset", label_column="label",
        class_names=[BARCELONA_KEY, MUNICH_KEY], num_classes=2, kind="binary",
        label_dtype=torch.float32, target_shape="vector1",
        default_loss="bce",
    ),
    "type": TaskSpec(
        name="type", label_column="label",
        class_names=FINAL_CLASSES, num_classes=10, kind="multiclass",
        label_dtype=torch.long, target_shape="scalar",
        default_loss="cross_entropy",
    ),
    "criticality": TaskSpec(
        name="criticality", label_column="label",
        class_names=CRITICALITY_CLASSES, num_classes=3, kind="multiclass",
        label_dtype=torch.long, target_shape="scalar",
        default_loss="cross_entropy",
    ),
    "binary_criticality": TaskSpec(
        name="binary_criticality", label_column="label",
        class_names=BINARY_CRITICALITY_CLASSES, num_classes=2, kind="multiclass",
        label_dtype=torch.long, target_shape="scalar",
        default_loss="cross_entropy",
    ),
}
```

Remarques :

- `dataset` reste traité comme une tâche `kind="binary"` avec BCE et une sortie à
  1 logit (comportement actuel de `DatasetClassifier` + `BCEWithLogitsLoss`).
- Le `num_classes` réel est recalculé depuis le DataFrame
  (`df[task.label_column].nunique()`) et **comparé** à `TaskSpec.num_classes` :
  un `warnings.warn` si un sous-échantillon (`n_images`) ne contient pas toutes
  les classes. Le modèle et les métriques sont dimensionnés sur
  `TaskSpec.num_classes` (dimension fixe, cohérente entre runs).
- `TaskSpec` est le seul objet transmis à `build_model`, `build_loss`,
  `build_metrics` et `BloodCellDataset`. Plus aucun branchement sur une string
  `mode` ailleurs.

## 5. Données et transformations

### `features/dataframe.py`

`create_dataframe(data_cfg: DataConfig, task: TaskSpec) -> pd.DataFrame` : reprend
`create_dataframe` + `make_update_data` de `features/dataset.py`. Le paramètre
`mode` devient `task` ; la colonne de stratification vient de `data_cfg.stratify`
(valeur explicite portée par la config, `"no"` = pas de stratification).

### `features/dataset.py` (slim)

```python
class BloodCellDataset(Dataset):
    def __init__(self, data: pd.DataFrame, task: TaskSpec, transform=None):
        self.data = data
        self.task = task
        self.transforms = transform

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        with Image.open(row.path) as img:
            image = v2.functional.pil_to_tensor(img.convert("RGB"))
        if self.transforms:
            image = self.transforms(image)
        label = torch.tensor(row[self.task.label_column], dtype=self.task.label_dtype)
        if self.task.target_shape == "vector1":
            label = label.unsqueeze(0)
        return image, label
```

`DatasetTransform` (wrapper appliquant un transform à un `Subset`) est conservé
tel quel. `DomainDataset` / `CriticalityDataset` sont supprimés.

### `features/stats.py`

- `compute_dataset_mean_std(train_data, transform_cfg, reinhard_normalizer=None)`
  déplacé depuis `features/dataset.py`.
- `average_reinhard_target_stats(...)` et `build_reinhard_normalizer(...)`
  déplacés depuis `features/transform.py`.
- `build_reinhard(train_df, transform_cfg) -> ReinhardNormalizer` : helper
  qui encapsule la séquence actuelle de `train.py:190-204`
  (instanciation `ReinhardNormalizer` + `average_reinhard_target_stats` sur un
  échantillon stratifié par dataset d'origine).

### `features/transform.py`

`build_transform` change de signature :

```python
def build_transform(
    cfg: TransformConfig,
    *,
    training: bool,
    reinhard_normalizer=None,
    dataset_mean=None,
    dataset_std=None,
) -> v2.Compose: ...
```

Le corps (ordre resize → jpeg → to-unit → augmentation géométrique →
Reinhard → CLAHE → jitter → normalize) est **conservé à l'identique** ; seuls les
littéraux sont remplacés par des champs de `cfg`. Les classes `ResizeWithPad`,
`ToUnitRange`, `ReinhardTransform`, `CLAHETransform` restent dans ce fichier.

## 6. Trainer

`src/training/trainer.py`. Le `Trainer` ne connaît ni DataFrame, ni split, ni
Reinhard : il reçoit des `Dataset` déjà transformés.

```python
class Trainer:
    def __init__(
        self, model: nn.Module, config: TrainConfig, criterion, *,
        optimizer, scheduler=None, metrics: MetricCollection | None = None,
        callbacks: Sequence = (), device: str | torch.device = "cpu",
    ): ...

    def fit(self, train_ds, val_ds=None) -> dict[str, list]:
        # DataLoader(train_ds, config.batch_size, shuffle=True) ; idem val_ds
        # garde-fou : (scheduler plateau ou callback early stopping) sans val_ds -> ValueError
        # pour epoch in range(config.nb_epoch):
        #     metrics_train = self._train_epoch(...)
        #     history["lr"].append(optimizer.param_groups[0]["lr"])
        #     metrics_val = self._validate_epoch(...) si val_loader
        #     history[k].append(v) pour k, v dans {**metrics_train, **metrics_val}
        #     scheduler.step(metrics_val["val_loss"]) si ReduceLROnPlateau, sinon scheduler.step()
        #     for cb in callbacks: if cb.step({**metrics_train, **metrics_val}, model): stop
        # for cb in callbacks: cb.restore(model)
        return history

    def evaluate(self, test_ds, metrics: MetricCollection | None = None) -> dict:
        # reprise de BaseModel.evaluate : DataLoader sans shuffle, running loss + metrics
        ...

    def predict(self, x):
        return self.model(x)
```

- `_train_epoch` / `_validate_epoch` : portage quasi littéral de
  `BaseModel._train_batches` / `validate_batches` (tqdm, running loss pondérée
  par la taille de batch, `MetricCollection` clonée avec préfixes `train_` /
  `val_`, `reset()` en fin d'epoch d'entraînement).
- **Callbacks** : le protocole est « objet exposant
  `step(epoch_metrics: dict, model: nn.Module) -> bool` et `restore(model)` ».
  `EarlyStopping` (déplacé de `models/callbacks.py` vers `training/callbacks.py`)
  garde son API **inchangée** (`monitor`, `mode`, `patience`, `min_delta`,
  `restore_best_weights`, attributs `best_score` / `best_epoch` / `stopped_epoch`).
- **La validation croisée sort du Trainer** : c'est `run_experiment` qui boucle
  sur les folds et instancie un `Trainer` neuf (modèle + optimizer + scheduler +
  callbacks neufs) par fold. Disparaissent de l'ancien `fit` : la branche `cv`,
  `reset_weights()`, `deepcopy(early_stopping)` par fold, la closure
  `transform_fn`.

### `training/components.py`

- `resolve_device(name) -> torch.device` : `"auto"` → `cuda` sinon `mps` sinon
  `cpu` (bloc de `train.py:235`).
- `split_off_test(df, data_cfg, task) -> tuple[pd.DataFrame, pd.DataFrame]` :
  `train_test_split(df, test_size=data_cfg.test_size, stratify=...)` sur la
  colonne `data_cfg.stratify` (reprend `train.py:186-188`).
- `make_splits(train_df, train_cfg, task) -> Iterator[tuple[pd.DataFrame, pd.DataFrame | None]]` :
  - `validation="none"` → un seul `(train_df, None)` ;
  - `validation="holdout"` → un seul `(tr, va)`, `va` de taille `val_size`,
    stratifié sur la colonne `stratify` ;
  - `validation="cross"` → `cross_val_folds` paires via `StratifiedKFold`.
- `TransformBuilder(transform_cfg, reinhard_normalizer)` avec
  `.build(fit_df: pd.DataFrame, *, training: bool) -> v2.Compose` : calcule
  `compute_dataset_mean_std` sur `fit_df` (le bon sous-ensemble d'entraînement,
  éventuellement le fold courant), puis appelle `build_transform`. Remplace la
  closure `transform_fn` de `train.py`. Aucune mémoïsation entre folds — les
  stats sont recalculées par fold.
- `build_optimizer(optim_cfg, params)` : `Adam` / `SGD` avec `lr`, `weight_decay`.
- `build_scheduler(sched_cfg, optimizer, *, nb_epoch)` : `None` / `ReduceLROnPlateau`
  (`mode="min"`, `patience`, `min_lr=lr_min`) / `CosineAnnealingLR`
  (`T_max=nb_epoch`, `eta_min=lr_min`) / `StepLR`
  (`step_size=max(1, nb_epoch // 3)`, `gamma=step_gamma`). Reprend
  `build_scheduler_factory` de `train.py:114`.

## 7. Runner d'expérience

`src/experiments/runner.py`.

```python
@dataclass
class RunResult:
    config: ExperimentConfig
    task: TaskSpec
    histories: list[dict]              # une entrée par fold (une seule en holdout/none)
    test_metrics: dict                 # scalaires ; en CV : {"f1_macro_mean", "f1_macro_std", ...}
    test_metrics_folds: list[dict]     # métriques test par fold (CV) ; [métriques] en holdout
    best_state: dict                   # state_dict du modèle retenu (fold 0 en CV)
    run_dir: Path
    duration_s: float
    status: Literal["ok", "failed"] = "ok"

def run_experiment(cfg: ExperimentConfig) -> RunResult:
    seed_everything(cfg.seed)
    task = get_task(cfg.task)
    df = create_dataframe(cfg.data, task)
    _warn_if_missing_classes(df, task)
    device = resolve_device(cfg.train.device)

    train_df, test_df = split_off_test(df, cfg.data, task)      # train_test_split stratifié
    normalizer = build_reinhard(train_df, cfg.transform) if cfg.transform.reinhard else None
    tb = TransformBuilder(cfg.transform, normalizer)

    test_ds = DatasetTransform(BloodCellDataset(test_df, task), tb.build(train_df, training=False))

    histories, fold_test_metrics, trained = [], [], []
    for fold, (tr_df, va_df) in enumerate(make_splits(train_df, cfg.train, task)):
        model     = build_model(cfg.model, task).to(device)
        criterion = build_loss(cfg.loss, task, tr_df).to(device)
        optimizer = build_optimizer(cfg.optim, model.parameters())
        scheduler = build_scheduler(cfg.scheduler, optimizer, nb_epoch=cfg.train.nb_epoch)
        callbacks = ([EarlyStopping(monitor="val_loss", mode="min",
                                    patience=cfg.train.early_stopping_patience)]
                     if cfg.train.early_stopping_patience else [])

        train_ds = DatasetTransform(BloodCellDataset(tr_df, task), tb.build(tr_df, training=True))
        val_ds = (DatasetTransform(BloodCellDataset(va_df, task), tb.build(tr_df, training=False))
                  if va_df is not None else None)

        trainer = Trainer(model, cfg.train, criterion, optimizer=optimizer, scheduler=scheduler,
                          metrics=build_metrics(task), callbacks=callbacks, device=device)
        histories.append(trainer.fit(train_ds, val_ds))
        fold_test_metrics.append(trainer.evaluate(test_ds, metrics=build_metrics(task, for_test=True)))
        trained.append(model)

    result = _assemble_result(cfg, task, histories, fold_test_metrics, trained)
    write_run_dir(result)
    return result
```

### Comportement par mode de validation

| `validation` | folds | figures epoch | éval test | leaderboard |
|---|---|---|---|---|
| `none` | 1 (train complet, pas de val) | courbes train seules | 1 modèle → métriques test | métriques test |
| `holdout` | 1 | courbes train + val | 1 modèle → métriques test | métriques val (best) + test |
| `cross` | N | **non tracées** (`history.json` conservé par fold) | **chaque fold évalué sur le même test** → moyenne ± écart-type | val agrégée (moy ± std) + test (moy ± std) |

- En CV, les figures test (matrice de confusion, barplot par classe) sont
  générées depuis le **fold 0** (une seule figure ; ni N figures, ni une somme
  trompeuse sur le même jeu de test). `best_state` = `state_dict` du fold 0.
- `display_metrics(...)` est appelé pour produire figures + CSV + résumé imprimé
  dans tous les cas où une éval test a lieu (c.-à-d. toujours).

## 8. Grid search

`src/experiments/grid.py`.

```python
def expand_grid(base: ExperimentConfig, grid_file: Path) -> list[ExperimentConfig]:
    raw = yaml.safe_load(Path(grid_file).read_text())   # {"loss.name": [...], "loss.beta": [...]}
    return [_apply_overrides(base, combo) for combo in ParameterGrid(raw)]

def run_grid(base: ExperimentConfig, grid_file: Path, *, task: str | None = None,
             resume: bool = False) -> None:
    if task:
        base = _apply_overrides(base, {"task": task})
    leaderboard = base.output.root / "leaderboard.csv"
    for cfg in expand_grid(base, grid_file):
        run_dir = base.output.root / dirname_for(cfg)
        if resume and run_dir.exists():
            continue
        try:
            append_leaderboard(leaderboard, run_experiment(cfg))
        except Exception as exc:                         # noqa: BLE001 — une combo qui casse ne stoppe pas le grid
            append_leaderboard(leaderboard, _failed_result(cfg, exc))
```

- `_apply_overrides(base, flat_dict)` réutilise **exactement** la même mécanique
  que `--set` : `flat_dict` (clés pointées) → dict imbriqué → `_deep_merge` sur
  `base.model_dump()` → `ExperimentConfig.model_validate(...)`. Une combinaison
  invalide lève une `ValidationError` avant tout entraînement.
- `ParameterGrid` accepte aussi une liste de dicts (grilles conditionnelles),
  utile pour ne pas croiser `loss.beta` quand `loss.name == "cross_entropy"`.

### `grids/*.yaml` (correspondance avec `reports/todo.md` §6)

```yaml
# configs/grids/preprocessing.yaml
transform.reinhard: [true, false]
transform.clahe:    [true, false]
transform.jpeg:     [true, false]
```

```yaml
# configs/grids/augmentation.yaml
transform.p_rotation:      [0.0, 0.5, 0.9]
transform.p_brightness:    [0.0, 0.5]
transform.p_contrast:      [0.0, 0.5]
transform.brightness_delta:[0.1, 0.2]
transform.contrast_delta:  [0.1, 0.2]
```

```yaml
# configs/grids/imbalance.yaml
- loss.name: [cross_entropy]
- loss.name: [focal]
  loss.beta: [0.99, 0.999, 0.9999]
  loss.gamma: [1.0, 2.0]
```

## 9. Artefacts

`src/experiments/artifacts.py`.

### Layout d'un run

```
results/<YYYYMMDD-HHMMSS>_<hash8>/
    config.yaml            # cfg figée, rechargeable telle quelle par load_config
    history.json           # dict unique (holdout/none) ou liste de dicts (CV, un par fold)
    metrics_test.json      # scalaires ; en CV : *_mean / *_std + liste par fold
    figures/
        training_loss.png   training_lr.png
        training_acc.png    training_f1.png            # selon les métriques de la tâche
        metrics_test_per_class.png
        metrics_test_confusion_matrix.png
        metrics_test_per_class.csv
        metrics_test_confusion_matrix.csv
    checkpoint.pt          # best_state, si output.save_checkpoint
```

- `<hash8>` = 8 premiers hex du SHA-256 du JSON canonique
  (`json.dumps(cfg.model_dump(mode="json"), sort_keys=True)`). Deux configs
  identiques → même dossier.
- Les figures epoch réutilisent `helpers.plot_train_metrics(history, save_dir)`
  **inchangé** ; en CV, non appelé.
- Les figures test réutilisent `metrics.display_metrics(results, num_classes,
  title="Test", save_dir=run_dir / "figures")` **inchangé**.

### `leaderboard.csv`

Une ligne par run, colonnes :

- `run_dir`, `timestamp`, `task`, `seed`, `duration_s`, `status`, `error`
- hyperparamètres aplatis : `flatten(cfg.model_dump())` →
  `data.n_images`, `transform.reinhard`, `loss.name`, `loss.beta`, `optim.lr`,
  `scheduler.name`, `train.nb_epoch`, … (préfixe = chemin pointé)
- métriques clés : `test_loss`, `f1_macro`, `mcc`, `min_critical_recall`
  (+ suffixes `_mean` / `_std` en CV), `val_f1_macro` (epoch de meilleure
  `val_loss`, holdout/CV)
- `min_critical_recall` = min du rappel sur les classes critiques
  (`CRITICAL_LABELS`) ; renseigné pour `task="type"` et `task="criticality"`,
  `NaN` pour les tâches sans notion de criticité (`dataset`, `binary_criticality`)

Append via `pandas` (`df.to_csv(mode="a", header=not exists)`). Tri / filtrage
hors-ligne.

## 10. CLI

```bash
# run unique
uv run python -m src.experiments.run  --config configs/base.yaml
uv run python -m src.experiments.run  --config configs/base.yaml \
    --set train.nb_epoch=10 loss.name=focal loss.beta=0.999

# grid search (une invocation = une tâche)
uv run python -m src.experiments.grid --config configs/base.yaml \
    --grid configs/grids/imbalance.yaml --task type
uv run python -m src.experiments.grid --config configs/base.yaml \
    --grid configs/grids/imbalance.yaml --task criticality --resume
```

- `run.py` : argparse `--config` (obligatoire), `--set KEY=VAL ...` (`nargs="*"`).
- `grid_cli.py` : `--config`, `--grid` (obligatoires), `--set ...`, `--task`,
  `--resume`.
- `--task` sur le grid surcharge `cfg.task` (raccourci pour « une invocation =
  une tâche »).
- `configs/base.yaml` reprend les valeurs par défaut de `ExperimentConfig`,
  écrites explicitement (documentation vivante).
- Les ~20 flags de `src/train.py` sont supprimés : ils vivent désormais dans
  `configs/base.yaml` ou passent par `--set`.

## 11. Tests

### Assemblage des builders

`config.py` reste sans import torch. Les constructeurs sont des fonctions libres,
appelées par le runner : `build_model` (`models/__init__.py`), `build_loss`
(`models/losses.py`), `build_metrics` (`models/metrics.py`), `build_optimizer` /
`build_scheduler` (`training/components.py`).

### Tests existants

| Fichier | Devenir |
|---|---|
| `tests/models/test_callbacks.py` | → `tests/training/test_callbacks.py`. Seul l'import change (`src.training.callbacks`). API `EarlyStopping` inchangée, assertions inchangées. |
| `tests/models/test_helpers.py` | Inchangé (`helpers.py` non touché). |
| `tests/models/test_train_loop.py` | → `tests/training/test_trainer.py`. Les 6 cas (scheduler 1×/epoch ; plateau reçoit val_loss ; lr dans history ; early stopping coupe + restaure ; early stopping / plateau sans val → `ValueError`) transposés contre `Trainer.fit(train_ds, val_ds)` avec `TinyModel(nn.Module)` + `TensorDataset`. |

### Tests neufs

- `tests/test_config.py` : charge `configs/base.yaml` ; `_parse_overrides`
  (`"train.lr=1e-3"` → `float`) ; `_deep_merge` ; clé inconnue → `ValidationError` ;
  `validation="none"` + `scheduler.name="plateau"` → `ValidationError`.
- `tests/test_tasks.py` : `get_task(x)` → `num_classes` / `class_names` /
  `label_dtype` / `target_shape` corrects pour les 4 tâches ; `get_task("xxx")`
  → `KeyError`.
- `tests/training/test_components.py` : `make_splits` (nombre de folds ;
  stratification préservée ; `None` en mode `none`) ; `build_scheduler` /
  `build_optimizer` (type + hyperparamètres) ; `TransformBuilder.build` →
  `v2.Compose`, `mean` / `std` finis sur un DataFrame minuscule.
- `tests/experiments/test_grid.py` : `expand_grid` = produit cartésien (taille,
  contenu) ; grille en liste de dicts ; `_apply_overrides` re-valide (combo
  invalide → `ValidationError`).
- `tests/experiments/test_artifacts.py` : `write_run_dir(fake_result, tmp_path)`
  crée `config.yaml` / `history.json` / `metrics_test.json` / `figures/*` ;
  `config.yaml` rechargeable par `load_config` ; `append_leaderboard` ajoute une
  ligne aux colonnes aplaties attendues, puis une deuxième sans réécrire l'entête.
- `tests/test_losses.py` : `build_loss(LossConfig(name="focal"), task_type, df)`
  → `ClassBalancedFocalLoss` de dimension `task.num_classes` ;
  `build_loss(LossConfig(), task_dataset, df)` → `BCEWithLogitsLoss`. Formalise le
  bloc `__main__` actuel de `losses.py`.
- `tests/experiments/test_runner_smoke.py` : `run_experiment` sur config
  minuscule (`n_images≈40`, `nb_epoch=1`, `device="cpu"`, `reinhard=False`,
  `clahe=False`, `jpeg=False`) → dossier de run peuplé + 1 ligne leaderboard.
  `@pytest.mark.skipif(not DATASET_PATHS[BARCELONA_KEY].exists(), reason="données absentes")`.

Commande : `/Users/romain/Projects/Liora/Projet/.venv/bin/python -m pytest tests -q`.

## 12. Séquence de build

1. `config.py` + `tests/test_config.py`.
2. `tasks.py` + `tests/test_tasks.py`.
3. Éclatement `features/dataset.py` → `features/dataframe.py` + `features/dataset.py`
   slim + `features/stats.py` ; `BloodCellDataset(df, task, transform)` ; adapter
   les imports appelants (`train.py` encore en place à ce stade).
4. `features/transform.py` → `build_transform(TransformConfig, *, training, ...)` ;
   déplacer `average_reinhard_target_stats` / `build_reinhard_normalizer` vers
   `features/stats.py` ; ajouter `build_reinhard(train_df, cfg)`.
5. `models/` : `train_model.py` → `baseline.py` (nn.Module pur) ;
   `dataset_model.py` sans `BaseModel` ; `models/__init__.py:build_model` ;
   `build_loss` dans `losses.py` ; `build_metrics(task, for_test=False)` dans
   `metrics.py` ; `tests/test_losses.py`.
6. `training/` : déplacer `callbacks.py` (+ redirection de test) ; écrire
   `components.py` (+ `tests/training/test_components.py`) ; écrire `trainer.py`
   en portant la boucle de `base_model.py` ; réécrire `tests/training/test_trainer.py` ;
   **supprimer `src/models/base_model.py`**.
7. `experiments/` : `artifacts.py` (+ test) → `runner.py` → `grid.py` (+ test) →
   `run.py` / `grid_cli.py` ; `tests/experiments/test_runner_smoke.py`.
8. `configs/base.yaml` + `configs/grids/{preprocessing,augmentation,imbalance}.yaml`.
9. **Supprimer `src/train.py`**.
10. `pytest tests -q` vert ; un run manuel de chaque CLI (run unique holdout +
    grid court sur 2-4 combinaisons).
11. Ajouter `pydantic>=2` aux dépendances (`pyproject.toml`) et installer dans le
    venv parent (`VIRTUAL_ENV=/Users/romain/Projects/Liora/Projet/.venv uv pip install 'pydantic>=2'`).

## 13. Hors périmètre

- **`WeightedRandomSampler` (todo #3)** : non implémenté. Point d'ajout unique
  plus tard — champ `DataConfig.sampler: Literal["none", "weighted"]` + branche
  dans la construction du `DataLoader` d'entraînement (`Trainer.fit`). Rien à
  défaire dans ce design.
- **Modèle transfer learning** : `build_model` lève
  `NotImplementedError("model 'transfer' : à implémenter")`. `ModelConfig.name`
  accepte déjà `"transfer"`, `TransferClassifier` viendra dans `models/transfer.py`.
- **Tracking (MLflow / W&B)** : non. `leaderboard.csv` + dossiers de run
  suffisent ; `RunResult` rend un adaptateur trivial à ajouter.
- **Optuna / recherche continue** : non. `ParameterGrid` couvre les axes
  catégoriels de `todo.md`.
- **Notebooks** (`notebooks/*.ipynb`) : non modifiés. Ceux qui importent
  `src.features.dataset.create_dataframe` ou `src.models.train_model` casseront
  sur les modules déplacés ; la liste des imports à corriger sera fournie, la
  correction est à la main de l'utilisateur.
- **`src/models/predict_model.py`, `src/features/build_features.py`** (vides),
  `src/streamlit/`, `src/visualization/` : non touchés.

## 14. Risques et points d'attention

- **Portage de la boucle d'entraînement** : `_train_epoch` / `_validate_epoch`
  doivent reproduire exactement la sémantique actuelle (loss moyennée pondérée
  par la taille de batch, `MetricCollection.reset()` au bon moment, `no_grad` en
  validation). Les tests `test_trainer.py` verrouillent le comportement du
  scheduler et de l'early stopping ; ajouter une assertion sur la valeur de
  `train_loss` pour un mini-batch déterministe.
- **Stats Reinhard coûteuses** : `build_reinhard` échantillonne 500 images.
  Acceptable pour un run, potentiellement lourd sur un grid `preprocessing`
  (`reinhard: [true, false]`). Pas d'optimisation prévue ; à surveiller, le cache
  éventuel viendra plus tard.
- **CV × grid = coût multiplicatif** : `cross_val_folds=3` triple chaque
  combinaison. Documenter dans `configs/base.yaml` que le grid tourne par défaut
  en `holdout`.
- **`dataset` (BCE, 1 logit)** : vérifier que `build_metrics` gère bien
  `task="dataset"` (`Accuracy(task="binary")`) et que `display_metrics` ne
  suppose pas systématiquement du multiclasse.
- **Dépendance pydantic v2** : nouvelle dépendance runtime (pas seulement dev).
  À ajouter à `pyproject.toml` **et** `requirements.txt`.
