---
title: "Note — rééquilibrage des classes : perte, matrice de coût et sampler"
author: "Romain"
date: "12 septembre 2026"
---

# Rééquilibrage des classes : perte, matrice de coût et sampler

**Note de travail** — à intégrer au §Rééquilibrage / Perte du rapport (suite de
la comparaison `grid_search_imbalance_1`, réalisée avec l'ancien système à
trois pertes distinctes, cf. `note_perte_unifiee.md`). Couvre les grilles
menées avec la perte unifiée `ClinicalLoss` sur les deux tâches `criticality`
et `type`, du 9 au 12 septembre 2026.

## Problématique

Le jeu de données est fortement déséquilibré, sur les deux nomenclatures
utilisées :

- **tâche `type`** (10 classes fines) : rapport effectif majoritaire/minoritaire
  **17,5** (Neutrophile ≈ 12 000 vs Promyélocyte 680) ;
- **tâche `criticality`** (3 paliers normal / intermédiaire / critique,
  agrégés depuis `type`) : déséquilibre plus modéré, rapport **≈ 6,9**
  (27 479 / 3 989 / 3 974).

Ce déséquilibre est particulièrement problématique ici parce que les classes
rares (Blast, Promyélocyte) sont aussi les classes **cliniquement critiques** :
un modèle qui optimise l'accuracy ou même le F1 macro global peut très bien
sacrifier le rappel sur ces classes sans que la métrique globale ne le
signale. Le pilotage se fait donc sur un couple de métriques — `f1_macro`
(performance globale) et `min_critical_recall` (le plus faible rappel parmi
les classes/paliers critiques) — dont on cherche le meilleur compromis
conjoint plutôt qu'à optimiser l'une au détriment de l'autre.

Trois leviers de rééquilibrage sont disponibles et peuvent en principe se
combiner :

1. **pondération de la perte** (`ClinicalLoss` : `beta` pour une pondération
   *class-balanced*, `gamma` pour un terme focal) ;
2. **rééchantillonnage** (`WeightedRandomSampler`, `train.sampler_alpha`) ;
3. **matrice de coût clinique** (`loss.lambda_cost` + `loss.cost_matrix`),
   qui code l'asymétrie du risque (rater un cas critique coûte plus cher
   qu'une fausse alerte) plutôt que la seule rareté statistique.

La question centrale de cette phase de travail : ces trois leviers
corrigent-ils le **même** déséquilibre (auquel cas les cumuler revient à le
compter plusieurs fois, au détriment du F1 global) ou sont-ils
complémentaires ? Et — question qui s'est révélée plus importante que prévu —
comment calibrer correctement la matrice de coût, jusque-là jamais réglée
finement ?

## Méthodologie

Pas de description exhaustive de chaque grille testée (`configs/grids/`
`criticality_loss.yaml`, `loss_matrix.yaml`, `imbalance.yaml`) — la démarche
générale :

1. Exploration initiale sur la tâche `criticality` (3 classes, entraînement
   rapide, ~4 min/run) : perte class-balanced + focale seule, puis ajout du
   terme de coût avec la **matrice par défaut**.
2. Réglage fin des **valeurs** de la matrice de coût (terme de coût isolé,
   `beta=0, gamma=0`), pour comprendre pourquoi le premier essai était décevant.
3. Combinaison de la matrice retenue avec le sampler et la pondération de
   classe, pour trancher la question de la redondance.
4. Généralisation à la tâche `type` (10 classes) : la matrice de coût y est
   *liftée* depuis l'espace de la criticité (`M_10[i,j] = M_3[crit(i), crit(j)]`,
   cf. `note_perte_unifiee.md`), avec le même protocole.

Métriques : `val_f1_macro`, `val_min_critical_recall`, et leur moyenne
`val_f1_recall` comme critère de synthèse pour classer les configurations.

**Précaution méthodologique** (cf. roadblocks ci-dessous) : toute
présélection sur un seed unique a été reconfirmée par un run **multi-seed
apparié** — les mêmes 2-3 seeds rejoués sur toutes les configurations
comparées, jamais un seed différent par configuration (qui biaiserait la
comparaison en faveur d'un tirage favorable).

## Résultats

### Tâche `criticality`

| Étape | Meilleure config | `f1_macro` | rappel critique min |
|---|---|---|---|
| Class-balanced + focal seuls, sans coût | `beta=0.9999, gamma=2-3` | 0,919 | 0,957 |
| Coût avec matrice **par défaut** | `lambda_cost>0` | ↓ (jusqu'à 0,88) | ≈ inchangé |
| **Coût avec matrice réglée** (retenue) | voir ci-dessous | **0,909** | **0,964** |

Avec la matrice par défaut, le terme de coût dégradait le F1 macro sans
améliorer le rappel critique — il semblait donc inutile. En réglant les
valeurs de la matrice (abaisser le coût `critique → normal` de 10 à 4,
`intermédiaire → normal` de 4 à 2, garder `critique → intermédiaire` à 3), le
terme de coût devient au contraire le **meilleur levier** de la tâche,
dépassant nettement le meilleur réglage class-balanced + focal.

Combiner en plus le sampler et la pondération de classe par-dessus cette
matrice n'apporte **pas de gain net** (confirmé multi-seed) : une fois la
matrice et le sampler actifs, `beta` devient redondant.

**Configuration retenue — tâche `criticality`** :

```yaml
loss:
  beta: 0.0
  gamma: 0.0
  lambda_cost: 1.0
  cost_matrix: [[0.0, 0.5, 1.0], [2.0, 0.0, 1.0], [4.0, 2.0, 0.0]]
train:
  sampler: none
```

→ `f1_macro ≈ 0,909 (σ ≈ 0,009)`, rappel critique `≈ 0,964 (σ ≈ 0,006)`,
contre `≈ 0,91 / 0,90` pour une cross-entropy nue (`beta=gamma=lambda_cost=0`).

### Tâche `type`

Ici, la conclusion s'inverse : **cumuler les trois leviers aide**. La matrice
de coût seule apporte un gain net par rapport à l'absence de matrice
(+4,7 points de rappel critique en moyenne, à F1 macro quasi inchangé), mais
il faut en plus :

- un sampler à **pleine intensité** (`sampler_alpha = 1.0`) — le déséquilibre
  de `type` (17,5) est trop sévère pour qu'un rééchantillonnage partiel
  suffise, contrairement à `criticality` ;
- une pondération de classe **non nulle** (`beta = 0.999`) — elle garde ici un
  effet mesurable en plus du sampler et de la matrice, contrairement à
  `criticality` où elle devenait redondante.

**Configuration retenue — tâche `type`** :

```yaml
loss:
  beta: 0.999
  gamma: 0.0
  lambda_cost: 1.0
  cost_matrix: [[0.0, 0.5, 1.0], [2.0, 0.0, 1.0], [4.0, 3.0, 0.0]]
train:
  sampler: weight
  sampler_alpha: 1.0
```

→ `f1_macro ≈ 0,871 (σ ≈ 0,007)`, rappel critique `≈ 0,910 (σ ≈ 0,024)`,
contre `≈ 0,867 / 0,845` pour la meilleure combinaison class-balanced + focal
+ sampler **sans** matrice de coût.

### À retenir

Les deux tâches n'ont **pas le même optimum** : la recette qui gagne sur
`criticality` (matrice seule) n'est pas celle qui gagne sur `type` (matrice +
sampler plein + pondération modérée). Le degré de rééquilibrage nécessaire
dépend de la sévérité du déséquilibre de la tâche, pas seulement de la
présence de classes critiques.

## Roadblocks rencontrés

**La matrice par défaut a failli enterrer la piste "coût clinique".**
Le premier essai (matrice par défaut, `lambda_cost > 0`) dégradait le F1 sans
gain de rappel — une conclusion hâtive aurait été « le terme de coût ne sert
à rien sur ce projet ». En réalité c'est la **calibration** de la matrice qui
était en cause (des coûts `→ normal` trop élevés poussaient le modèle à
sur-prédire les paliers supérieurs, dégradant la précision sans gagner de
rappel) : une fois réglée, le terme de coût est devenu le meilleur levier.
Leçon : un mécanisme qui ne marche pas avec ses valeurs par défaut n'est pas
forcément un mécanisme inutile.

**Une erreur de saisie sur la matrice s'est révélée être un vrai paramètre.**
`M[critique, critique]` avait été fixé à 1 au lieu de 0 par erreur. Le
correctif (remettre 0) n'a pas simplement corrigé un bug sans conséquence :
en multi-seed apparié, `M[critique,critique]=1` s'est avéré agir comme un
frein contre la sur-prédiction du palier critique, utile quand le reste de la
matrice pousse fort vers ce palier, contre-productif sinon. Les deux réglages
ont donc été comparés explicitement plutôt que de supposer que « 0 est
forcément mieux que 1 ».

**Le seed d'exploration était un tirage favorable.**
Comparé après coup à deux autres seeds (comparaison appariée), le seed
utilisé pour l'exploration initiale surestimait le F1 macro d'environ
1,3 point par rapport à la moyenne des trois seeds, avec un écart-type
intra-configuration d'environ 0,009 sur `f1_macro`. Conséquence pratique :
les écarts inférieurs à ~0,01 point observés dans les grilles à seed unique
n'étaient pas interprétables, et tout classement resserré (« top 3 » d'une
grille) a dû être rejoué en multi-seed avant d'être considéré comme acquis.

**Le bruit inter-seed est beaucoup plus élevé sur la tâche `type`.**
Un premier grid combinant matrice de coût et sampler sur `type` n'avait, par
construction, que 4 des 12 matrices testées avec les trois seeds — les 8
autres n'avaient qu'un seul run. Le rappel critique s'est avéré varier
jusqu'à 0,18 entre seeds pour une même configuration (les classes critiques
fines, ex. Promyélocyte, ne représentent qu'une centaine d'images en
validation : l'estimation de leur rappel est mécaniquement instable sur un
échantillon aussi petit). Ce grid a dû être refait de zéro, correctement
apparié (mêmes 3 seeds sur les 47 configurations), avant de pouvoir trancher
quoi que ce soit sur `type`.

## Suite

- Les deux configurations retenues ci-dessus sont celles à figer pour les
  runs finaux (test set, une seule fois) qui alimenteront les résultats du
  rapport.
- Sur `type`, l'espace autour de la matrice retenue (`critique→interm = 3`)
  n'a pas été exploré aussi finement que sur `criticality` — matrices
  voisines encore à tester si le temps le permet.
