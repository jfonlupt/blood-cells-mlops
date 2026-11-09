---
title: "Difficultés et pièges rencontrés — journal technique"
author: "Romain"
date: "12 septembre 2026"
---

# Difficultés et pièges rencontrés

**Document de travail — projet Blood Cell Classification**

> Journal cumulatif des difficultés techniques, bugs et pièges rencontrés au
> cours du projet, au moment où on les rencontre. Chaque entrée documente un
> problème concret : symptôme, diagnostic, cause racine, correction et leçon à
> en tirer. L'objectif est de ne rien perdre de ce travail d'investigation
> pour la rédaction finale du rapport (section « Difficultés rencontrées » ou
> équivalent) — copier/résumer les entrées pertinentes plutôt que de repartir
> de zéro.
>
> Convention : une entrée par difficulté, la plus récente en premier, avec un
> titre court, la date, les fichiers concernés et un statut
> (`résolu` / `contournement` / `ouvert`).

---

## Sommaire

- [Comparaisons de grid search faussées par un seed unique non apparié](#seed-variance-rebalancing) — résolu, 9–12 septembre 2026
- [Un `1` au lieu d'un `0` dans la matrice de coût : erreur de saisie devenue hyperparamètre](#cost-matrix-diagonal-typo) — résolu, 9–12 septembre 2026
- [Grid search « matrice de coût par défaut » jugé inutile — c'était un problème de calibration](#cost-matrix-default-miscalibrated) — résolu, 9–12 septembre 2026
- [Perte à trois implémentations distinctes : hyperparamètres silencieusement ignorés selon `loss.name`](#loss-silent-ignored-params) — résolu, 9 septembre 2026
- [Leaderboard de grid search inutilisable après la correction de la fuite test](#val-rich-metrics) — résolu, 4 septembre 2026
- [`best_state` en cross-validation : premier fold au lieu du meilleur](#best-fold-cv) — résolu, 4 septembre 2026
- [Fuite du jeu de test dans la sélection de modèle du grid search](#leak-test-grid-search) — résolu, 4 septembre 2026
- [Collapse de l'entraînement avec `num_workers` > 0 (absence de gradient clipping)](#collapse-num-workers) — résolu, 4 septembre 2026

---

## Comparaisons de grid search faussées par un seed unique non apparié {#seed-variance-rebalancing}

**Date :** 9–12 septembre 2026
**Statut :** résolu
**Fichiers concernés :** `configs/grids/criticality_loss.yaml`, `configs/grids/loss_matrix.yaml`, `configs/grids/imbalance.yaml`

### Contexte

Exploration des grilles de rééquilibrage (pondération de perte, matrice de coût
clinique, sampler) sur les tâches `criticality` (3 paliers) et `type` (10 classes),
menée à seed unique dans un premier temps pour aller vite (~4 min/run sur
`criticality`).

### Symptôme

1. Rejoué après coup sur deux seeds supplémentaires, le seed d'exploration
   initial surestimait le F1 macro d'environ **1,3 point** par rapport à la
   moyenne des trois seeds, avec un écart-type intra-configuration d'environ
   **0,009** sur `f1_macro` — les écarts inférieurs à ~0,01 point observés
   entre configurations d'une même grille à seed unique n'étaient donc pas
   interprétables.
2. Sur la tâche `type`, un premier grid combinant matrice de coût et sampler
   n'avait, par construction, que 4 des 12 matrices testées avec les trois
   seeds (les 8 autres n'avaient qu'un seul run). Le rappel critique s'est
   avéré varier jusqu'à **0,18** entre seeds pour une même configuration.

### Cause racine

Les classes critiques fines (ex. Promyélocyte) ne représentent qu'une centaine
d'images en validation : l'estimation de leur rappel est mécaniquement
instable sur un échantillon aussi petit, et un tirage de seed favorable peut
suffire à faire gagner artificiellement une configuration lors d'une
comparaison à seed unique. Comparer des lignes de grille avec un seed
différent chacune (plutôt que les mêmes seeds rejoués sur toutes les
configurations comparées) biaise en plus la comparaison en faveur des tirages
favorables de certaines lignes.

### Correction

Règle méthodologique adoptée pour la suite du projet : toute présélection
faite sur un seed unique est reconfirmée par un run **multi-seed apparié**
(les mêmes 2–3 seeds rejoués sur toutes les configurations comparées, jamais
un seed différent par configuration). Le grid `type` incomplet a été refait de
zéro, correctement apparié (mêmes 3 seeds sur les 47 configurations), avant de
pouvoir trancher quoi que ce soit sur cette tâche.

### Leçon

- Sur des classes rares (peu d'exemples en validation), la variance
  inter-seed d'une métrique par classe peut dépasser largement l'écart entre
  configurations qu'on cherche à départager : un classement à seed unique,
  même reproductible en apparence (le même run rejoué donne le même
  résultat), n'est pas fiable en dessous d'un certain écart.
- Un multi-seed mal construit (seeds différents selon la ligne du grid, ou
  seulement une partie des lignes rejouée) donne une fausse impression de
  robustesse statistique tout en biaisant la comparaison : l'appariement
  (mêmes seeds, sur toutes les configurations comparées) n'est pas
  négociable.
- Explorer large à seed unique puis reconfirmer étroit en multi-seed est un
  compromis raisonnable sous budget de calcul limité — mais un classement à
  seed unique ne doit jamais être considéré comme acquis en dessous d'un
  delta de l'ordre de l'écart-type inter-seed observé (~0,01 sur `f1_macro`
  ici).

---

## Un `1` au lieu d'un `0` dans la matrice de coût : erreur de saisie devenue hyperparamètre {#cost-matrix-diagonal-typo}

**Date :** 9–12 septembre 2026
**Statut :** résolu (comparé explicitement plutôt que corrigé aveuglément)
**Fichiers concernés :** `configs/grids/loss_matrix.yaml`, `src/models/losses.py`

### Contexte

En réglant finement les valeurs de la matrice de coût clinique `cost_matrix`
(cf. l'entrée [« matrice de coût par défaut jugé inutile »](#cost-matrix-default-miscalibrated)
ci-dessous), le coefficient
diagonal `M[critique, critique]` — censé être nul par construction (coût nul
à prédire correctement) — avait été saisi à `1` par erreur dans une des
grilles testées.

### Symptôme

Repéré à la relecture de la matrice dans la config, pas par un résultat
aberrant : rien dans les métriques ne signalait le problème.

### Cause racine (et pourquoi ce n'était pas qu'un bug anodin)

Le réflexe naturel aurait été de corriger silencieusement (remettre `0`, la
valeur « évidemment correcte » au sens de la spécification d'une matrice de
coût). Mais en comparaison multi-seed appariée, `M[critique,critique]=1`
s'est révélé agir comme un **frein contre la sur-prédiction du palier
critique** : utile quand le reste de la matrice pousse fort vers ce palier
(fortes pénalités hors-diagonale sur les autres lignes), contre-productif
sinon. Ce n'était donc pas une erreur sans conséquence sur le comportement du
modèle.

### Correction

Les deux réglages (`0` et `1`) ont été comparés explicitement en multi-seed
plutôt que de supposer que la valeur mathématiquement « propre » est
forcément la meilleure empiriquement ; le choix retenu dépend du reste de la
matrice plutôt que d'être fixé une fois pour toutes.

### Leçon

- Une valeur « évidemment fausse » au sens de la spécification (ici, une
  diagonale non nulle dans une matrice censée en avoir une nulle) peut malgré
  tout avoir un effet réel et mesurable une fois injectée dans un système
  d'optimisation — la corriger sans vérifier revient à perdre un réglage
  potentiellement utile sans même savoir qu'il existait.
- Face à ce genre de trouvaille, comparer explicitement les deux versions sur
  les mêmes seeds plutôt que de trancher par intuition ou par pure cohérence
  théorique.

---

## Grid search « matrice de coût par défaut » jugé inutile — c'était un problème de calibration, pas de mécanisme {#cost-matrix-default-miscalibrated}

**Date :** 9–12 septembre 2026
**Statut :** résolu
**Fichiers concernés :** `src/models/losses.py` (`ClinicalLoss`, terme de coût espéré), `configs/grids/loss_matrix.yaml`

### Contexte

Sur la tâche `criticality` (3 paliers), premier essai du terme de coût
clinique (`lambda_cost > 0`) avec la matrice de coût par défaut, héritée de
l'ancienne `CostSensitiveCriticalityLoss` (cf. `rapport_fusion.md`) :
`critique→normal = 10`, `intermédiaire→normal = 4`, `critique→intermédiaire = 3`.

### Symptôme

Le F1 macro se dégradait nettement (jusqu'à 0,88, contre 0,919 pour
class-balanced + focal seuls, sans terme de coût), sans amélioration mesurable
du rappel critique. Conclusion hâtive possible à ce stade : « le terme de coût
ne sert à rien sur ce projet ».

### Cause racine

Les coûts `→ normal` de la matrice par défaut étaient trop élevés pour ce
jeu de données : le modèle sur-prédisait les paliers critique et intermédiaire
pour éviter la pénalité, ce qui dégradait la précision sur ces classes sans
gagner de rappel côté critique (déjà élevé par ailleurs sans terme de coût).

### Correction

Réglage manuel des valeurs de la matrice (abaisser `critique→normal` de 10 à
4, `intermédiaire→normal` de 4 à 2, garder `critique→intermédiaire` à 3) :
le terme de coût devient alors le **meilleur levier** de la tâche
(`f1_macro ≈ 0,909`, rappel critique `≈ 0,964`, contre `≈ 0,91 / 0,90` pour
une cross-entropy nue) — nettement au-dessus du meilleur réglage
class-balanced + focal seul.

### Leçon

- Une matrice de coût, comme tout hyperparamètre porteur d'une hypothèse
  forte (ici l'asymétrie clinique du risque), doit être calibrée sur le jeu
  de données réel : une valeur par défaut raisonnable *a priori* peut être
  largement sur-dimensionnée pour un déséquilibre ou un contexte particulier.
- Avant de conclure qu'un mécanisme « ne marche pas », isoler son effet
  (ici : couper `beta` et `gamma` à 0 pour observer le terme de coût espéré
  seul) et balayer ses propres valeurs plutôt que de le juger sur un unique
  réglage par défaut.

---

## Perte à trois implémentations distinctes : hyperparamètres silencieusement ignorés selon `loss.name` {#loss-silent-ignored-params}

**Date :** 9 septembre 2026
**Statut :** résolu
**Fichiers concernés :** `src/models/losses.py`, `src/config.py`, `configs/*.yaml`, `configs/grids/criticality_loss.yaml`

### Contexte

Avant unification, trois classes de perte distinctes coexistaient dans
`src/models/losses.py`, sélectionnées par `loss.name` (`cross_entropy`,
`focal`, `cost_sensitive`), chacune ne consommant qu'un sous-ensemble des
champs de `LossConfig`.

### Symptôme

Aucune erreur ni warning : `gamma` (terme focal) n'avait aucun effet avec
`cross_entropy` ou `cost_sensitive` ; `lambda_cost` (coût espéré) n'avait
aucun effet avec `cross_entropy` ou `focal`. Repéré en relisant
`configs/grids/criticality_loss.yaml`, où deux blocs de grille faisaient
varier `gamma` et `lambda_cost` sans que ces valeurs aient le moindre effet
sur la loss réellement instanciée par ces blocs — des lignes de grille
différentes produisant des runs strictement identiques.

### Cause racine

`build_loss()` instanciait une classe différente selon `loss.name`. Chaque
classe n'exposait dans son constructeur que les champs de `LossConfig` qui la
concernaient ; le reste du dataclass restait présent (donc lisible,
modifiable en config, journalisé dans le leaderboard) sans jamais être lu.

### Correction

Fusion des trois pertes en une seule classe `ClinicalLoss` (§ voir
`note_perte_unifiee.md`) combinant les trois termes — CE pondérée
*class-balanced*, terme focal, coût espéré — dans une seule formule :

$$\mathcal{L}_i = -\,w_y\,(1-p_y)^{\gamma}\,\log p_y \;+\; \lambda \sum_c M[y,c]\,p_c$$

chaque terme désactivable en mettant son coefficient à une valeur neutre
(`beta=0`, `gamma=0`, `lambda_cost=0`) plutôt que par sélection
d'implémentation. Le champ `loss.name` est supprimé de `LossConfig` : il
n'existe plus de branche capable d'ignorer un champ existant.

### Leçon

- Plusieurs implémentations sélectionnées par un champ `name`/`type`,
  partageant un même bloc de configuration, sont un terrain propice aux
  hyperparamètres silencieusement ignorés : dès qu'un champ n'est pas
  consommé par la branche active, rien ne le signale (pas d'erreur, pas de
  warning, la valeur reste visible et crédible dans les logs).
- Signe d'alerte concret à surveiller dans un grid search : des lignes qui
  font varier un paramètre censé influencer le résultat mais produisent des
  runs strictement identiques.
- Fusionner en une seule formule paramétrique, où chaque terme se désactive
  par sa propre valeur neutre, élimine la classe de bug plutôt que de la
  corriger config par config.

---

## Leaderboard de grid search inutilisable après la correction de la fuite test {#val-rich-metrics}

**Date :** 4 septembre 2026
**Statut :** résolu
**Fichiers concernés :** `src/training/trainer.py`, `src/experiments/runner.py`, `src/experiments/artifacts.py`

### Contexte

Suite directe de [la correction de la fuite du jeu de test](#leak-test-grid-search) :
en désactivant l'évaluation test pour le grid search, le leaderboard ne
gardait plus que `val_f1_macro` (calculé légèrement, depuis l'historique
d'entraînement) — toutes les métriques riches (`mcc`, per-class, `top3`, et
surtout `min_critical_recall`, un critère spec important pour ce projet de
santé) devenaient `NaN` pour chaque run de grille.

### Symptôme

Remarque directe : « le grid search ajoute une ligne dans le leaderboard sans
métriques ! c'est impossible à utiliser ».

### Tentation (écartée) : réintroduire l'évaluation test dans le grid search

La demande initiale était de faire évaluer un jeu de test par le grid search.
Mais le faire sur **le même** test set que celui réservé au run final aurait
tout simplement réintroduit la fuite qu'on venait de corriger : chaque
combinaison serait de nouveau comparée sur le même test set.

### Correction

Calculer le même jeu de métriques riches (`build_metrics(task, for_test=True)`)
sur le **jeu de validation** plutôt que sur le test set — le val set est déjà
légitimement utilisé pour comparer des configs, ce n'est pas une fuite :

- `Trainer.evaluate()` généralisé : accepte un paramètre `prefix` (`"test"`
  par défaut, `"val"` pour la validation) au lieu de coder en dur le préfixe
  `test_`. `_flatten_metric()` idem.
- `run_experiment()` : évalue systématiquement `val_ds` avec la collection
  riche (`prefix="val"`), en plus de `trainer.fit()` — toujours, pas
  seulement quand `evaluate_test=False`, pour un comportement cohérent entre
  run unique et grid search.
- `RunResult` : nouveaux champs `val_metrics` / `val_metrics_raw` (miroir de
  `test_metrics` / `test_metrics_raw`).
- `append_leaderboard()` : nouvelles colonnes `val_min_critical_recall`,
  `val_mcc`, `val_top3_*`, etc. — `val_f1_macro` est maintenant écrasé par la
  version riche (recalculée) si présente. `_min_critical_recall()` généralisé
  avec un paramètre `prefix`.
- `write_run_dir()` : nouveau `metrics_val.json` + figures « Val » (matrice de
  confusion, per-class) en plus de « Test » quand elles existent.

Vérifié sur un run réel (200 images, 1 epoch, grid 1 combinaison) : la ligne
de leaderboard a bien `min_critical_recall=NaN` (test jamais touché) et
`val_min_critical_recall`, `val_f1_macro`, `val_mcc`, `val_top3_acc`,
`val_top3_f1` tous renseignés.

### Leçon

- Corriger une fuite de données ne doit pas se faire en supprimant purement
  et simplement l'évaluation concernée : il faut retrouver un jeu **légitime**
  (ici, la validation) pour porter le même besoin (comparer des
  configurations), sinon la correction casse un usage réel.
- Une fonction qui code en dur un préfixe de clés (``"test_"``) au lieu de le
  paramétrer se referme sur un seul cas d'usage — la généraliser
  (`prefix="test"` par défaut) a permis de la réutiliser pour la validation
  sans dupliquer `Trainer.evaluate()`.

---

## `best_state` en cross-validation : premier fold au lieu du meilleur {#best-fold-cv}

**Date :** 4 septembre 2026
**Statut :** résolu
**Fichiers concernés :** `src/experiments/runner.py`

### Contexte

En relisant `run_experiment()` : chaque fold de `make_splits()` entraîne un
modèle et l'ajoute à une liste `models`, puis `RunResult.best_state` était
construit avec `models[0].state_dict()`.

### Symptôme

Aucun crash — repéré à la lecture du code (« je ne vois pas où `models` est
trié »). En `validation="none"`/`"holdout"`, `make_splits()` ne produit
qu'**un seul** fold, donc `models[0]` est trivialement correct : le bug est
invisible tant qu'on ne fait pas de `validation="cross"`.

### Cause racine

En `validation="cross"` (`cross_val_folds` folds, 3 par défaut), `models`
contient un modèle par fold, mais rien ne les compare : `models[0]` est
simplement le modèle du *premier* fold entraîné, pas le meilleur. Le nom
`best_state` était trompeur.

Sous-problème lié (non corrigé, à garder en tête) : `models[i]` correspond
aux poids de la **dernière epoch** du fold `i`, pas forcément sa meilleure —
sauf si `train.early_stopping_patience > 0`, auquel cas `EarlyStopping`
restaure déjà la meilleure epoch *à l'intérieur* de ce fold
(`src/training/callbacks.py`). "Best" n'est donc garanti qu'*à l'intérieur*
d'un fold (et seulement avec early stopping actif), jamais *entre* folds
avant cette correction.

### Correction

Nouvelle fonction `_best_fold_index(histories)` : compare le `val_loss`
minimal (sur les epochs) de chaque fold, et renvoie l'index du fold gagnant
— même critère que `_val_f1_macro` déjà utilisé pour le leaderboard. Fallback
sur l'index 0 si un fold n'a pas de `val_loss` (`validation="none"`, ou un
seul fold). `best_state=models[_best_fold_index(histories)].state_dict()`.

### Leçon

- Une liste remplie dans une boucle (`models.append(model)`) sans étape de
  tri/sélection explicite est un piège classique : le premier élément est
  toujours utilisable syntaxiquement (`[0]`), donc rien ne signale l'absence
  de logique de sélection — ni erreur ni warning.
- Ce genre de bug reste invisible tant qu'on ne teste que le chemin par
  défaut (ici holdout, un seul fold) : à surveiller spécifiquement quand un
  paramètre de config change le nombre d'itérations d'une boucle (holdout vs
  cross-validation).

---

## Fuite du jeu de test dans la sélection de modèle du grid search {#leak-test-grid-search}

**Date :** 4 septembre 2026
**Statut :** résolu
**Fichiers concernés :** `src/experiments/runner.py`, `src/experiments/grid.py`

### Contexte

`run_experiment()` (utilisé aussi bien par un run unique, `src.experiments.run`,
que par le grid search, `src.experiments.grid`) sépare un split test au début
(`split_off_test`, stratifié, seedé par `config.seed`) et l'évaluait
**systématiquement** en fin de run (`trainer.evaluate(test_ds, ...)`), avec le
résultat écrit dans `results/leaderboard.csv`.

### Symptôme

Aucun crash ni résultat aberrant — un problème méthodologique silencieux,
repéré en se demandant simplement : « est-ce que le grid search utilise le
jeu de test ? ».

### Cause racine

Aucune des grilles de `configs/grids/` ne fait varier `data.*` ni `seed` :
chaque combinaison d'un grid search reçoit donc **exactement le même split
test**. Comme `run_grid()` appelle `run_experiment()` pour chaque combinaison
et que celle-ci évaluait toujours le test set, **toutes les combinaisons
étaient comparées entre elles via leur score sur le même test set**, et ce
score atterrissait dans le leaderboard servant à choisir la meilleure
combinaison. Le test set ne fuite pas dans l'entraînement des poids (chaque
run entraîne bien sur `train`/`val` uniquement), mais il fuite dans la
**sélection de modèle** : choisir la ligne du leaderboard avec le meilleur
`test_f1_macro` revient à optimiser indirectement sur le test set, comme s'il
s'agissait d'un jeu de validation. Les métriques test du run "gagnant" d'une
grille étaient donc optimistes.

### Correction

- `run_experiment()` : nouveau paramètre `evaluate_test: bool = True`. À
  `False`, l'évaluation test est sautée (`test_metrics={}`,
  `test_metrics_raw=None`), sans casser les runs uniques (défaut inchangé).
- `run_grid()` : appelle désormais `run_experiment(..., evaluate_test=False)`
  — la comparaison entre combinaisons se fait uniquement sur `val_f1_macro`
  (déjà présent dans le leaderboard, calculé à l'epoch de `val_loss`
  minimale).
- Workflow propre : choisir la meilleure combinaison sur `val_f1_macro` dans
  le leaderboard du grid search, puis relancer cette config précise via
  `src.experiments.run` (qui évalue le test set une seule fois, sur le
  modèle final retenu).
- Suite (même jour) : construire `test_ds` n'est pas gratuit —
  `tb.build(train_df, training=False)` déclenche un vrai passage de sonde
  (`compute_dataset_mean_std`, ~640 images prétraitées) pour rien si
  `evaluate_test=False`. `test_ds` n'est donc plus construit du tout dans ce
  cas.

### Leçon

- Toute réutilisation d'un même split de test pour comparer plusieurs
  configurations constitue une fuite dans la sélection de modèle, même sans
  toucher aux poids du réseau. Le réflexe : le test set ne doit être touché
  **qu'une fois**, sur la configuration finale déjà choisie.
- Les runs de grid search déjà présents dans `results/leaderboard.csv` avant
  cette correction ont potentiellement ce biais — à garder en tête si on les
  réutilise pour le rapport final (comparer plutôt sur `val_f1_macro` a
  posteriori, ou relancer les meilleures combinaisons sans fuite).

---

## Collapse de l'entraînement avec `num_workers` > 0 (absence de gradient clipping) {#collapse-num-workers}

**Date :** 4 septembre 2026
**Statut :** résolu
**Fichiers concernés :** `src/training/trainer.py`, `src/config.py`, `configs/base.yaml`

### Contexte

Ajout d'un paramètre `train.num_workers` pour paralléliser le chargement des
données (`DataLoader`) et accélérer l'entraînement sur Mac (Apple Silicon,
15 cœurs). Un premier souci de latence (~5 s avant que la barre de progression
n'avance à chaque epoch) a été résolu simplement avec
`persistent_workers=True` (macOS respawn les processus workers à chaque
epoch par défaut, ce qui est coûteux avec `spawn`, la méthode de
multiprocessing utilisée par défaut sur macOS — contrairement à Linux qui
utilise `fork`).

### Symptôme

Avec `train.num_workers=8` sur le run de référence (`configs/base.yaml`,
tâche `type`, 15 000 images), l'entraînement restait bloqué à
l'accuracy du hasard (`train_acc≈0.10`, `val_acc≈0.10` sur 10 classes) du
début à la fin des 10 epochs — alors que la même commande avec
`train.num_workers=0` entraînait normalement (`val_acc` de 0.56 à 0.69 sur
les mêmes 10 epochs, même seed).

### Fausses pistes explorées (et pourquoi elles ont été écartées)

1. **Corruption des données par le multiprocessing** — vérifié par script
   dédié : labels intacts, pas de NaN, statistiques d'image cohérentes,
   shuffle qui change bien à chaque epoch, avec ou sans workers. Écarté.
2. **Jeu de données trop petit** — un test à `n_images=1000` collapsait aussi
   bien avec `num_workers=0` qu'avec `num_workers=8` : confondant (l'
   instabilité existe indépendamment du nombre de workers à cette échelle),
   pas révélateur du vrai problème.
3. **CLAHE (kornia) instable sous multiprocessing** — désactiver CLAHE
   (`transform.clahe=false`) n'empêchait pas le pic de loss initial, seulement
   la sévérité du collapse. Piste partiellement correcte mais pas la cause
   racine.
4. **Différence de flux d'augmentation aléatoire entre workers** — chaque
   worker `DataLoader` reçoit sa propre graine RNG (PyTorch reseed
   automatiquement chaque process), donc la séquence de rotations/flips/jitter
   diffère de celle du process principal, même pour les mêmes images dans le
   même ordre (le shuffle, lui, est déterminé dans le process principal).
   Confirmé comme *facteur déclenchant* (trajectoire d'entraînement
   différente), mais désactiver l'augmentation ne suffisait pas non plus à
   empêcher le pic de loss — seulement à permettre la récupération après coup.

### Cause racine

Le pic de loss initial (repérable en isolant le log par batch : la moyenne
courante de `train_loss` saute de 2.3 à ~11-12 au 2ᵉ batch, ce qui correspond
à une perte réelle d'environ 21 sur ce seul batch) est en réalité un
événement plausible dès les premiers batches d'un entraînement from-scratch
avec Adam à `lr=0.01` sur un CNN sans BatchNorm. Le vrai bug n'était pas ce
pic en lui-même, mais l'absence totale de **gradient clipping** dans
`Trainer._train_epoch` : rien n'empêchait ce gradient exceptionnellement grand
de détruire les poids en une seule mise à jour. Le réseau restait ensuite
bloqué à prédire la classe majoritaire pour le reste de l'entraînement, sans
jamais s'en remettre.

`train.num_workers` n'est donc pas la cause du bug — il change simplement la
trajectoire aléatoire de l'entraînement (via les graines RNG indépendantes
par worker), ce qui a suffi, avec ce seed précis, à déclencher un batch
initial anormal. Le même risque existe en principe avec `num_workers=0` sur
un seed différent ; il était resté invisible jusqu'ici.

### Diagnostic — méthode

Ce qui a permis de trancher : reproduire le run exact (même config, même
seed) en ne faisant varier qu'un seul paramètre à la fois (`num_workers`,
puis `transform.clahe`, puis les probabilités d'augmentation), en isolant à
chaque fois le log **par batch** (pas seulement le résumé de fin d'epoch, qui
moyenne et masque un pic ponctuel) pour repérer où exactement la trajectoire
divergeait. La confirmation finale : ajouter `clip_grad_norm_(max_norm=1.0)`
à titre de test sur le run qui collapsait auparavant → apprentissage normal
retrouvé (`test_f1_macro` passe de 0.02 à 0.34 sur un run d'1 epoch,
`num_workers=8` inchangé).

### Correction

- `src/training/trainer.py` : `torch.nn.utils.clip_grad_norm_(...)` avant
  `optimizer.step()`, appliqué seulement si `train.grad_clip_norm > 0`.
- `src/config.py` / `configs/base.yaml` : nouveau paramètre
  `train.grad_clip_norm` (activé par défaut à `1.0`).

### Leçon

- Le gradient clipping est une pratique standard et peu coûteuse pour
  sécuriser un entraînement from-scratch sans BatchNorm avec un `lr` élevé —
  son absence était un risque latent indépendant de `num_workers`, qui a
  seulement servi de déclencheur.
- Un run qui « marche » une fois avec un seed donné ne garantit rien : il
  peut être simplement chanceux. Changer un paramètre a priori neutre
  (`num_workers`) peut suffire à changer la trajectoire aléatoire et révéler
  une fragilité déjà présente.
- Pour diagnostiquer un collapse d'entraînement, toujours inspecter la
  courbe **par batch** de la première epoch (pas seulement les métriques de
  fin d'epoch, qui lissent/masquent les pics ponctuels).

---

<!--
Gabarit pour une nouvelle entrée — copier ce bloc en tête de section
« Collapse... » ci-dessus (les entrées les plus récentes en premier) et
compléter :

## Titre court du problème {#ancre-slug}

**Date :**
**Statut :** résolu / contournement / ouvert
**Fichiers concernés :**

### Contexte

### Symptôme

### Cause racine

### Correction

### Leçon
-->
