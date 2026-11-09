---
title: "Note — unification de la fonction de perte"
author: "Romain"
date: "9 septembre 2026"
---

# Unification de la fonction de perte

**Note de travail** — à intégrer au §Loss du rapport plus tard.

## Avant

Trois pertes distinctes, choisies par `loss.name` :

- `cross_entropy` → `nn.CrossEntropyLoss` nue ;
- `focal` → `ClassBalancedFocalLoss` (pondération *class-balanced* + terme focal) ;
- `cost_sensitive` → `CostSensitiveCriticalityLoss` (CE pondérée + coût espéré), réservée aux tâches `criticality` / `binary_criticality`.

Défaut : `loss.name` posait aussi quels hyperparamètres étaient lus — `gamma` était
silencieusement ignoré par `cross_entropy` et `cost_sensitive`, `lambda_cost` par
`cross_entropy` et `focal`.

## Après — une seule perte

`src/models/losses.py::ClinicalLoss`. Pour un exemple de vraie classe $y$,
$p = \operatorname{softmax}(z)$ :

$$\mathcal{L}_i = \underbrace{-\,w_y\,(1-p_y)^{\gamma}\,\log p_y}_{\text{CE pondérée / focal / lissée}} \;+\; \underbrace{\lambda \sum_c M[y,c]\,p_c}_{\text{coût espéré}}$$

- $w_c$ : nombre effectif d'échantillons (Cui et al. 2019), réglé par `beta`
  ($0$ = non pondéré, $1$ = inverse-fréquence), × `extra_weight` (boost clinique
  manuel), renormalisé à moyenne 1 ;
- $(1-p_y)^{\gamma}$ : terme focal (Lin et al. 2017), `gamma = 0` = désactivé ;
- $\sum_c M[y,c]\,p_c$ : coût espéré, `lambda_cost = 0` = désactivé.

| Réglage | Retombe sur |
|---|---|
| `gamma=0, lambda_cost=0, beta=0` | cross-entropy nue |
| `beta > 0` | CE *class-balanced* |
| `gamma > 0` | focal |
| `lambda_cost > 0` (+ `cost_matrix`) | perte sensible au coût |

`nn.BCEWithLogitsLoss` reste utilisé pour les tâches à sortie 1 logit (`dataset`).

### Config (`LossConfig`)

`loss.name` **supprimé**. Champs : `beta` (borne `[0, 1]` incluse), `gamma`,
`lambda_cost`, `extra_weight`, `label_smoothing` (nouveau, exposé), `cost_matrix`.

### Matrice de coût liftée

`cost_matrix` (dans la config) vit dans l'espace de la **criticité** (3×3 par
défaut : `normal` / `intermediaire` / `critique` ; ou 2×2 : `normal` / `abnormal`).

- `task = criticality` / `binary_criticality` → utilisée telle quelle ;
- `task = type` (10 classes) → **liftée** : $M_{10}[i,j] = M_{k}[\,\text{crit}(c_i),\ \text{crit}(c_j)\,]$
  où $\text{crit}(\cdot)$ est le palier de criticité de la classe (`src/constants.py::CRITICALITY`).
  Conséquence : coût **nul** entre deux classes de même criticité (Blast ↔ Promyélocyte),
  coût maximal pour « critique vu comme normal ». Le terme CE porte la
  discrimination fine à l'intérieur d'un bloc de criticité.
- Une matrice fournie déjà à la dimension de la tâche (10×10) est utilisée directement.

Validation : `LossConfig` vérifie que la matrice est carrée et d'ordre ≥ 2 ;
`build_loss` vérifie que sa dimension est 2, 3, ou `num_classes` selon la tâche.

## Fichiers touchés

`src/models/losses.py` (réécrit), `src/config.py` (`LossConfig`),
`src/experiments/artifacts.py` (colonne leaderboard, cf. ci-dessous),
`src/experiments/run.py` / `grid.py` (textes d'aide), tests
(`test_losses.py` réécrit, `test_config.py`, `test_grid.py`).

## Migration

Configs migrées pour **préserver le comportement effectif** :

| Fichier | Avant | Après |
|---|---|---|
| `base.yaml`, `transfer.yaml` | focal β=0.99 γ=1 (`lambda_cost` ignoré) | `beta:0.99 gamma:1.0 lambda_cost:0.0` |
| `criticite.yaml`, `archi_model.yaml` | CE nue | `beta:0.0 gamma:0.0 lambda_cost:0.0` |
| `dataset.yaml` | BCE (params ignorés) | params neutres |
| `grids/imbalance.yaml` | `loss.name:[focal]` + β/γ | ligne `loss.name` retirée |

- **`configs/grids/criticality_loss.yaml`** : laissé tel quel (`loss.name` encore
  présent → invalide en l'état), à refaire manuellement. Les deux blocs y
  balayaient `gamma` / `lambda_cost` qui étaient sans effet ; avec la perte unique
  ils deviennent actifs.
- **Leaderboard** : `loss.name` retiré de la config → la colonne du CSV est
  conservée et remplie de `"clinical"` pour les nouveaux runs (les anciennes
  lignes gardent `focal` / `cost_sensitive` / `cross_entropy`). Les colonnes
  `loss.gamma`, `loss.beta`, `loss.lambda_cost`… portent l'information.

## Références

- Cui et al. 2019, *Class-Balanced Loss Based on Effective Number of Samples* (déjà en biblio).
- Lin et al. 2017, *Focal Loss for Dense Object Detection* (déjà en biblio).
