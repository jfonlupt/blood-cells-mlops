> **Copie publique** du projet d'équipe *Classification de cellules sanguines*
> (formation Data Scientist, DataScientest), réalisé avec Romain Hild et Rémi Launay.
> Instantané de la branche `main` au commit `e1af880` (09/11/2026).
> Les identifiants d'infrastructure (compte AWS, ressources) ont été remplacés par
> des valeurs d'exemple ; les jeux de données ne sont pas inclus (versionnés avec DVC).

Classification des cellules sanguines
=====================================

Pipeline d'entraînement et d'évaluation, piloté par configuration, pour la
classification d'images de cellules sanguines issues de la fusion de deux jeux
de données de frottis sanguins :

- **PBC** (Barcelone, `barcelona`) — 17 092 images de cellules de sujets sains ;
- **LMU** (Munich, `munich`) — 18 365 images de patients atteints ou non de
  leucémie myéloïde aiguë.

Soit **35 457 images** au total. Les deux sources diffèrent par leur
colorimétrie, leur résolution et leur nomenclature, et les classes sont
fortement déséquilibrées (long-tail, rapport ~17:1 sur les types principaux) —
le pipeline traite explicitement ces deux points (harmonisation couleur,
pondération de perte, rééchantillonnage).

Quatre tâches de classification, sélectionnées par `task` dans la config :

| `task`               | Classes | Description |
|----------------------|---------|-------------|
| `type`               | 10      | Type cellulaire (Basophil, Blast, Eosinophil, Erythroblast, IG, Lymphocyte, Monocyte, Neutrophil, Platelet, Promyelocyte) |
| `criticality`        | 3       | Criticité clinique : `normal` / `intermediaire` / `critique` |
| `binary_criticality` | 2       | Détection d'anomalie : `normal` / `abnormal` |
| `dataset`            | 2       | Source d'acquisition (`barcelona` / `munich`) — contrôle : un score élevé signale que le modèle apprend la source plutôt que la morphologie |

Une expérience = un fichier de config YAML (surchargeable en ligne de commande).
Chaque run produit un dossier d'artefacts identifié par le hash de sa config et
une ligne dans un leaderboard cumulatif.

Organisation du dépôt
---------------------

    ├── configs/                 <- Configs d'expérience (YAML)
    │   ├── base.yaml            <- Config de référence (défauts de ExperimentConfig)
    │   ├── dataset.yaml         <- Config dédiée à la tâche « dataset »
    │   ├── transfer.yaml        <- Config de référence pour le transfer learning (backbone timm)
    │   └── grids/               <- Grilles de recherche (une liste de surcharges par fichier)
    │       ├── preprocessing.yaml
    │       ├── augmentation.yaml
    │       ├── imbalance.yaml
    │       ├── dataset.yaml
    │       └── architecture/    <- Grilles d'architecture (dont transfer.yaml : comparaison de backbones)
    │
    ├── data/                    <- Images (hors git) : raw/ (brut) et processed/
    ├── docs/                    <- Specs et plans de conception
    ├── notebooks/               <- Notebooks d'exploration et de génération des figures du rapport
    ├── references/              <- Dictionnaires de données, manuels, liens
    ├── reports/                 <- Rapport LaTeX, todo, exploration
    │   └── figures/             <- Figures du rapport
    │
    ├── results/                 <- Sorties du pipeline (hors git)
    │   ├── leaderboard.csv      <- Une ligne par run (métriques + config aplatie)
    │   └── runs/<hash>/         <- Artefacts d'un run (cf. « Sorties d'un run »)
    │
    ├── src/
    │   ├── config.py            <- Arbre pydantic ExperimentConfig + chargement YAML + overrides CLI
    │   ├── constants.py         <- Classes, mappings de labels, criticité clinique
    │   ├── tasks.py             <- TaskSpec : labels, nombre de classes, loss par défaut d'une tâche
    │   ├── features/            <- DataFrame des images, Dataset PyTorch, transforms, stats (Reinhard, mean/std)
    │   ├── models/              <- Modèles (baseline, multi_archi, transfer, dataset), losses, métriques, Grad-CAM
    │   ├── training/            <- Trainer (boucle), callbacks (early stopping), components (build_optimizer/scheduler/sampler…)
    │   ├── experiments/         <- CLI (run.py, grid_cli.py), runner.py, grid.py, artifacts.py
    │   ├── visualization/       <- Visualisations exploratoires
    │   └── streamlit/           <- Application Streamlit
    │
    ├── tests/                   <- pytest (structure en miroir de src/)
    ├── pyproject.toml           <- Manifeste : dépendances du projet
    ├── uv.lock                  <- Résolution figée (gérée par uv)
    └── requirements.txt         <- Alternative pip (pip freeze de l'environnement)

Installation
------------

Python ≥ 3.14, [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Crée `.venv/` et installe toutes les dépendances déclarées dans `pyproject.toml`
(résolues dans `uv.lock`), y compris `pytest` et `ruff`. Préfixer ensuite les
commandes par `uv run`, ou activer l'environnement avec
`source .venv/bin/activate`.

Sans uv : `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

Les images doivent être présentes dans `data/` (hors git).

Entraînement
------------

**Run unique** — `--config` obligatoire, `--name` optionnel (nom lisible du run ;
un nom aléatoire type `brave-otter` est tiré si absent), `--set` pour surcharger
n'importe quelle clé de la config :

```
uv run python -m src.experiments.run --config configs/base.yaml
uv run python -m src.experiments.run --config configs/base.yaml --name baseline-focal \
    --set loss.gamma=2.0 loss.lambda_cost=1.0 train.nb_epoch=15
uv run python -m src.experiments.run --config configs/base.yaml --name sampler-sqrt \
    --set train.sampler=weight train.sampler_alpha=0.5
```

**Grid search** (une invocation = une tâche, exécution séquentielle) — `--grid`
pointe une grille de surcharges, `--task` force la tâche, `--name` est partagé
par **tous** les runs de la grille, `--resume` saute les runs déjà présents :

```
uv run python -m src.experiments.grid_cli --config configs/base.yaml \
    --grid configs/grids/imbalance.yaml --task type --name imbalance-sweep
uv run python -m src.experiments.grid_cli --config configs/base.yaml \
    --grid configs/grids/preprocessing.yaml --name preproc-sweep --resume
```

`configs/base.yaml` liste toutes les clés avec leurs valeurs par défaut
(prétraitement Reinhard/CLAHE/JPEG, augmentations, modèle, loss, optimiseur,
scheduler, validation holdout/cross, sampler…). Une grille est soit un dict de
listes (produit cartésien), soit une liste de tels dicts (union) ;
`grid.max_combinations` transforme la grille en recherche aléatoire.

### Transfer learning

`model.name: transfer` utilise un backbone pré-entraîné [timm](https://github.com/huggingface/pytorch-image-models)
(`model.transfer.backbone`, ex. `resnet50`, `efficientnet_b0`, `vit_base_patch16_224.augreg2_in21k_ft_in1k`)
suivi d'une tête MLP. Liste des backbones : `timm.list_models(pretrained=True)`
ou le [Hub](https://huggingface.co/models?library=timm&sort=trending)
([tableaux de perfs](https://huggingface.co/docs/timm/results)). Réglages sous `model.transfer` : `pretrained`,
`freeze` (`none` / `all` = *linear probing* / entier `N` = ne dégeler que les `N`
derniers groupes de stages), `head_hidden`, `head_dropout`, `backbone_lr_mult`
(taux d'apprentissage réduit du backbone). Avec un backbone pré-entraîné, mettre
`transform.norm_stats: imagenet`. Les poids sont téléchargés du Hugging Face Hub
au premier usage (puis mis en cache).

`configs/transfer.yaml` sert de référence ; `configs/grids/architecture/transfer.yaml`
compare 7 backbones (linear probing + fine-tuning) :

```
uv run python -m src.experiments.grid_cli --config configs/transfer.yaml \
    --grid configs/grids/architecture/transfer.yaml --name transfer-sweep
```

Re-Entraînement
------------

**Promote** (continue l'entraînement de modèles existants) — `--from-epochs`
reprend les modèles après un nombre d'epochs donné, `--top` sélectionne les meilleurs
runs, `--epochs` indique le nombre d'epochs supplémentaires et `--resume` saute les
runs déjà présents :

```
uv run python -m src.experiments.promote_cli \
    --from-epochs 8 --top 20 --epochs 3 --resume
```

**Retrain** (réentraîne les modèles depuis zéro avec de nouveaux paramètres) — `--from-epochs`
indique le nombre d'epochs utilisé pour sélectionner les modèles de référence,
`--retrain` pointe vers une configuration de réentraînement, `--top` sélectionne les
meilleurs runs et `--resume` saute les runs déjà présents. Le réentraînement permet
notamment de modifier les paramètres liés aux données, comme le nombre d'images :

```
uv run python -m src.experiments.retrain_cli \
    --from-epochs 20 --retrain configs/retrain.yaml --top 20 --resume
```

Où vont les résultats
---------------------

Chaque run écrit dans **`results/runs/<hash>/`** — le `<hash>` (8 caractères) est
dérivé de la config, stable et déterministe : relancer la même config réécrit le
même dossier. Chaque run ajoute une ligne à **`results/leaderboard.csv`**
(jamais réécrit).

Le `--name` sert à retrouver les runs d'une même expérience dans le
leaderboard — en particulier, tous les runs d'un même grid search partagent le
nom passé (ou tiré) au lancement.

### Sorties d'un run — `results/runs/<hash>/`

| Fichier | Contenu |
|---------|---------|
| `config.yaml`        | Config complète figée du run (rechargeable telle quelle) |
| `meta.json`          | `name`, `timestamp`, `status` |
| `history.json`       | Métriques par epoch — dict en holdout, liste de dicts en validation croisée (une entrée par fold) |
| `metrics_test.json`  | Métriques de test scalaires (agrégées `mean`/`std` en validation croisée) |
| `preprocessing.json` | Moyenne / écart-type du dataset + stats cible Reinhard — permet de reconstruire le transform d'évaluation sans relire les données |
| `checkpoint.pt`      | `state_dict` du modèle entraîné (premier fold en validation croisée) — si `output.save_checkpoint` |
| `figures/`           | PNG + CSV — si `output.save_figures` (voir ci-dessous) |

Contenu de `figures/` :

- `training_loss.png`, `training_lr.png`, `training_<métrique>.png`
  (`acc`, `f1`) — courbes d'entraînement/validation par epoch (holdout
  uniquement) ;
- `metrics_test_confusion_matrix.png` + `.csv` — matrice de confusion (comptes
  bruts + normalisée par ligne) ;
- `metrics_test_per_class.png` + `.csv` — rappel / précision / F2 / AUPRC (+ rappel top-3 si ≥ 5 classes) par classe ; le `.csv` ajoute le `support` (nombre d'images vraies par classe) ;
- `metrics_test_pr_curve.png` — courbe précision-rappel par classe, AP annotée.

### `results/leaderboard.csv`

Une ligne par run : `run_dir`, `name`, `timestamp`, `task`, `seed`,
`duration_s`, `status`, `error`, `min_critical_recall`, `val_f1_macro`, puis la
**config aplatie** (`loss.gamma`, `train.sampler`, `transform.jpeg_quality`, …)
et les **métriques de test** (`test_f1_macro`, `test_mcc`, …). Les colonnes sont
alignées par nom : des runs de tâches ou de modes de validation différents
cohabitent dans le même fichier.

Inférence & Grad-CAM
--------------------

Les deux CLI rechargent un run existant (`config.yaml` + `checkpoint.pt`) et
reconstruisent le prétraitement d'évaluation. Chemin rapide via
`preprocessing.json` (aucune donnée requise) ; pour les runs antérieurs à cette
sortie, la chaîne complète est rejouée depuis le `seed` (moyenne/écart-type
ré-estimés, pas au bit près).

**Inférence** — prédit la classe d'une ou plusieurs images :

```
uv run python -m src.experiments.infer --run results/runs/<hash> \
    --image a.jpg b.jpg --out predictions/
```

Affiche `image -> classe (proba)` et écrit `predictions/predictions.json` si `--out`.

**Grad-CAM** — carte de saillance sur une image, quatre PNG dans `--out` (créé
si besoin) : `<image>_original.png`, `<image>_heatmap.png`,
`<image>_overlay.png`, et `<image>_gradcam.png` (les trois côte à côte + barre
de couleurs).

```
uv run python -m src.experiments.gradcam --run results/runs/<hash> \
    --image a.jpg --out gradcam/ [--layer model.6] [--target-class Blast]
```

`--layer` : nom de couche (`model.named_modules()`), défaut = dernière `Conv2d`
pour un CNN maison, dernière conv spatiale du backbone pour un modèle `transfer`
(les backbones ViT / Swin ne sont pas couverts par le défaut).
`--target-class` : nom ou indice de classe à expliquer, défaut = classe prédite.

Métriques
---------

Fabriquées par `build_metrics(task, for_test=…)` (`src/models/metrics.py`).

**Par epoch** (entraînement / validation), légères :

| Métrique | Description |
|----------|-------------|
| `acc` | Accuracy |
| `f1`  | F1 macro (tâches multiclasses) |

**Sur le jeu de test** (multiclasses), riches :

| Métrique | Description |
|----------|-------------|
| `f1_macro`            | F1 macro (jamais l'accuracy seule, biaisée par Neutrophil) |
| `mcc`                 | Coefficient de corrélation de Matthews |
| `recall_per_class`    | Rappel par classe |
| `precision_per_class` | Précision par classe |
| `fbeta2_per_class`    | F-beta (β = 2, pondère le rappel 2× la précision) par classe |
| `auprc_per_class`     | Average precision (aire sous la courbe précision-rappel) par classe — nécessite les probabilités |
| `pr_curve`            | Courbe précision-rappel par classe → figure `metrics_test_pr_curve.png` |
| `confusion_matrix`    | Matrice de confusion complète |
| `top3_acc`            | Top-3 accuracy (micro) — vraie classe dans les 3 mieux notées. Seulement si `num_classes ≥ 5` |
| `top3_f1`             | Top-3 F1 macro — idem, borné à `num_classes ≥ 5` |
| `top3_recall_per_class` | Rappel top-3 par classe — idem |

Métriques dérivées écrites dans le leaderboard :

| Colonne | Description |
|---------|-------------|
| `min_critical_recall` | Rappel minimum sur les classes cliniquement critiques (`type` → Blast, Promyelocyte, … ; `criticality` → classe `critique`). Seuil visé : ≥ 0,95 |
| `val_f1_macro`        | `val_f1` à l'epoch de `val_loss` minimale |

**Tâche binaire** (`dataset`) : `acc` uniquement.

Tests
-----

```
uv run pytest -q
```

--------

<p><small>Structure initiale basée sur le <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>