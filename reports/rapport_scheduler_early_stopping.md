---
title: "Learning rate scheduling et early stopping — stratégies et implémentation"
author: "Romain"
date: "2 septembre 2026"
---

# Learning rate scheduling et early stopping

**Ébauche de rapport — projet Blood Cell Classification**

> Document de travail. Il décrit les stratégies de planification du taux
> d'apprentissage (*learning rate scheduler*) et d'arrêt anticipé (*early stopping*)
> retenues, les problèmes qu'elles résolvent et leur fonctionnement. L'implémentation
> correspondante se trouve dans `src/models/callbacks.py` (classe `EarlyStopping`),
> `src/models/base_model.py` (`BaseModel.fit` / `_train_and_validate`) et
> `src/train.py` (arguments CLI).

---

## 1. Contexte et problèmes à résoudre

L'entraînement d'un réseau de neurones par descente de gradient stochastique est piloté
par deux quantités qui déterminent en grande partie la qualité du résultat :

1. **le taux d'apprentissage** (`learning rate`, noté $\eta$) : l'amplitude du pas de
   mise à jour des poids à chaque *batch* ;
2. **le nombre d'epochs** : combien de fois le modèle voit l'ensemble du jeu
   d'entraînement.

Ces deux quantités posent chacune un problème pratique.

### 1.1 Le dilemme du taux d'apprentissage

Un taux d'apprentissage **fixe** est un compromis qui n'est bon qu'à un moment de
l'entraînement :

- **trop grand** : les mises à jour « sautent » par-dessus les minima de la fonction de
  perte, la perte oscille ou diverge, et le modèle ne se stabilise jamais dans une
  bonne solution ;
- **trop petit** : la convergence est lente, l'entraînement peut rester coincé dans un
  plateau ou un minimum local médiocre, et le budget de calcul est gaspillé.

En pratique, on veut un **grand pas au début** (explorer rapidement, sortir des zones
plates) puis un **petit pas à la fin** (affiner, se poser proprement au fond de la
vallée de perte). C'est exactement le rôle d'un *scheduler* : faire **décroître** $\eta$
au cours de l'entraînement selon une règle définie.

> **Interaction avec Adam.** Le projet utilise l'optimiseur Adam
> (`src/train.py`, `optimizer_factory`). Adam adapte déjà un pas par paramètre à partir
> des moments du gradient, ce qui atténue la sensibilité au choix de $\eta$. Mais Adam
> **ne réduit pas** le pas **global** en fin d'entraînement : la mise à jour reste de
> l'ordre de $\eta$ tant que les gradients ne s'annulent pas. Un scheduler global reste
> donc utile pour la phase de convergence fine, et c'est une pratique standard même
> avec Adam.

### 1.2 Le dilemme du nombre d'epochs

Fixer le nombre d'epochs à l'avance pose le problème du **surapprentissage**
(*overfitting*) :

- **trop peu d'epochs** : le modèle est sous-entraîné, il n'a pas exploité tout le
  signal disponible (*underfitting*) ;
- **trop d'epochs** : passé un certain point, la perte d'entraînement continue de
  baisser mais la perte de **validation remonte** — le modèle mémorise le bruit du jeu
  d'entraînement au lieu d'apprendre des motifs généralisables.

Le nombre optimal d'epochs dépend du modèle, des données, du preprocessing, de la loss,
du batch size… : il n'est pas connu d'avance et change à chaque configuration testée.
L'**early stopping** résout ce problème : arrêter l'entraînement automatiquement quand
la performance de validation cesse de progresser, et conserver les poids de la
**meilleure** epoch.

### 1.3 Enjeu spécifique au projet

Le jeu d'entraînement est la fusion de deux datasets hétérogènes (Barcelone / Munich)
avec un **fort déséquilibre de classes** et un **biais de domaine** marqué
(cf. `reports/rapport_transformation_augmentation.md`). Deux conséquences :

- la `val_loss` peut être **bruitée** (classes rares peu représentées dans le *split*
  de validation), ce qui rend les critères de plateau et d'arrêt plus délicats à
  régler ;
- le futur *grid search* (cf. `reports/todo.md`, point 6) va enchaîner de nombreuses
  configurations : l'early stopping devient indispensable pour ne pas fixer
  manuellement `nb_epoch` pour chacune, et pour borner le coût total.

---

## 2. Learning rate schedulers

### 2.1 Principe commun et intégration

Un scheduler PyTorch est un objet construit **autour de l'optimiseur** ; on appelle sa
méthode `step()` **une fois par epoch**, après l'étape d'optimisation, ce qui met à jour
le champ `param_groups[i]["lr"]` de l'optimiseur.

Dans le projet, le scheduler est instancié via une **factory**
`scheduler_factory(optimizer) -> scheduler` passée à `BaseModel.fit`. Ce choix est
imposé par le fait que l'optimiseur est créé **à l'intérieur** de `fit` (et recréé à
chaque *fold* en validation croisée) : une factory garantit que le scheduler est
toujours lié au bon optimiseur (`src/models/base_model.py`).

La boucle d'entraînement (`_train_and_validate`) applique la règle suivante à chaque
fin d'epoch :

```python
if isinstance(scheduler, ReduceLROnPlateau):
    scheduler.step(val_loss)   # a besoin de la métrique surveillée
else:
    scheduler.step()           # planning aveugle, piloté par le numéro d'epoch
```

Le taux d'apprentissage courant est enregistré dans `history["lr"]` et tracé par
`plot_train_metrics` (`figures/training_lr.png`).

### 2.2 `ReduceLROnPlateau` — réduction sur stagnation

**Problème résolu.** On ne sait pas *quand* réduire $\eta$. Plutôt que de le décider a
priori, on le décide **en observant les données** : tant que la métrique de validation
s'améliore, on ne touche à rien ; dès qu'elle stagne, on réduit.

**Fonctionnement.** À chaque epoch, le scheduler reçoit la valeur surveillée (ici
`val_loss`, `mode="min"`). Il maintient la meilleure valeur vue. Si pendant `patience`
epochs consécutives il n'y a **pas d'amélioration** supérieure à `threshold`, alors :

$$\eta \leftarrow \max(\eta \times \text{factor},\ \text{min\_lr})$$

puis il observe une période de `cooldown` epochs avant de recommencer à compter. Par
défaut `factor = 0.1` (division par 10), `threshold = 10^{-4}` en mode relatif.

Configuration dans le projet : `ReduceLROnPlateau(opt, mode="min",
patience=scheduler_patience, min_lr=lr_min)` — exposé par `--scheduler_patience` et
`--lr_min`.

| ✅ Avantages | ❌ Inconvénients |
|---|---|
| Piloté par la performance réelle, pas par un calendrier arbitraire | Réagit **après coup** : il faut stagner `patience` epochs avant d'agir |
| Aucune connaissance préalable du nombre d'epochs | Sensible au **bruit** de la `val_loss` (petit *split* de validation, classes rares) |
| Robuste en phase d'exploration | `factor = 0.1` est une division brutale : peut « éteindre » l'apprentissage si `patience` est faible |
| Même signal que l'early stopping → cohérence | Nécessite un jeu de validation |

**Quand l'utiliser.** Phase d'exploration, quand on ne sait pas combien d'epochs sont
nécessaires. À combiner avec l'early stopping (voir §4).

### 2.3 `CosineAnnealingLR` — décroissance cosinus

**Problème résolu.** On veut une décroissance **lisse et sans à-coup** de $\eta$, qui
termine à une valeur quasi nulle pour une convergence fine, sans avoir à choisir des
paliers.

**Fonctionnement.** Sur un horizon de `T_max` epochs, le taux suit une demi-période de
cosinus entre $\eta_0$ et $\eta_{\min}$ :

$$\eta_t = \eta_{\min} + \frac{1}{2}\,(\eta_0 - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi\, t}{T_{\max}}\right)\right)$$

La décroissance est lente au début (on reste proche de $\eta_0$), rapide au milieu,
puis de nouveau lente à l'approche de $\eta_{\min}$.

Configuration dans le projet : `CosineAnnealingLR(opt, T_max=nb_epoch, eta_min=lr_min)`
— l'horizon est donc calé sur `--nb_epoch`.

| ✅ Avantages | ❌ Inconvénients |
|---|---|
| Décroissance douce, aucune chute brutale | Exige de **connaître `nb_epoch` à l'avance** ($= T_{\max}$) |
| Fin d'entraînement à très petit $\eta$ → excellent score final, standard en vision | Mal couplé à l'early stopping : si on coupe à l'epoch 20/50, le planning n'a pas fini de descendre |
| Peu d'hyperparamètres ($T_{\max}$, $\eta_{\min}$) | Complètement **aveugle** aux données (planning figé) |
| Reproductible | Si `T_max` ≠ epochs réellement exécutées, le planning est incohérent |

**Variante.** `CosineAnnealingWarmRestarts` relance périodiquement $\eta$ à $\eta_0$
(« redémarrages chauds ») pour ré-explorer : utile pour les entraînements longs, non
retenu ici car nos runs sont courts.

**Quand l'utiliser.** Run « final » avec budget d'epochs fixé, sans early stopping
serré (ou avec une grande `patience` comme simple garde-fou).

### 2.4 `StepLR` — paliers fixes

**Problème résolu.** Recette simple et reproductible quand on a déjà une intuition du
moment où $\eta$ doit baisser.

**Fonctionnement.** Le taux est divisé par un facteur `gamma` tous les `step_size`
epochs :

$$\eta_t = \eta_0 \times \gamma^{\lfloor t / \text{step\_size} \rfloor}$$

Configuration dans le projet : `StepLR(opt, step_size=max(1, nb_epoch // 3),
gamma=0.1)` — soit trois paliers sur la durée de l'entraînement.

| ✅ Avantages | ❌ Inconvénients |
|---|---|
| Simple, prévisible, reproductible | Chutes **brutales** : sauts nets visibles dans la courbe de perte |
| Recette classique éprouvée (ResNet, etc.) | Deux hyperparamètres sensibles (`step_size`, `gamma`) |
| Bon si on connaît déjà le bon calendrier | Dépend de `nb_epoch` (via `step_size`) |
| | Aveugle aux données ; timing arbitraire si mal réglé |

**Variante.** `MultiStepLR` permet de placer les paliers à des epochs précises
(`milestones=[30, 45]`) plutôt qu'à intervalle régulier.

**Quand l'utiliser.** Reproduction d'une configuration connue, ou après avoir observé
empiriquement où la perte stagne.

### 2.5 `none` — taux constant

Référence de comparaison. Sur des entraînements très courts (les 3 epochs par défaut de
`--nb_epoch`), un scheduler n'apporte quasiment rien : la valeur ajoutée d'un planning
se manifeste sur plusieurs dizaines d'epochs. À conserver comme *baseline* dans le grid
search.

### 2.6 Autres schedulers (non implémentés)

- **`LambdaLR` / `OneCycleLR`** : `OneCycleLR` (Smith, 2018) fait *monter* puis
  *descendre* $\eta$ sur un seul cycle et est souvent le plus performant, mais il se
  pilote **par itération** (pas par epoch) et impose de connaître le nombre total de
  *steps* — incompatible avec la boucle actuelle sans modification.
- **Warmup** : montée linéaire de $\eta$ sur les premières epochs, utile pour les gros
  batchs et les *transformers* / le *transfer learning*. Pertinent si le modèle
  `transfer` (cf. `--model transfer`, non implémenté) est ajouté plus tard.

---

## 3. Early stopping

### 3.1 Problème résolu

Éviter le surapprentissage **et** éviter de fixer `nb_epoch` à la main pour chaque
configuration. On fixe une borne haute généreuse (`--nb_epoch 50` par exemple) et on
laisse l'early stopping couper au bon moment.

Allure typique : la perte d'entraînement décroît de façon monotone, la perte de
validation décroît puis atteint un minimum et **remonte**. Le bon modèle est celui du
minimum de validation, pas celui de la dernière epoch.

![Exemple de courbes de perte train / validation](figures/training_loss.png)

### 3.2 Fonctionnement (`src/models/callbacks.py`)

```python
EarlyStopping(monitor="val_loss", mode="min", patience=5,
              min_delta=0.0, restore_best_weights=True)
```

À chaque fin d'epoch, `_train_and_validate` appelle `early_stopping.step(epoch_metrics,
model)` :

1. **Lecture de la métrique** `epoch_metrics[monitor]` (par défaut `val_loss`). Si la
   clé est absente (pas de validation), une erreur explicite est levée.
2. **Test d'amélioration**, selon `mode` et `min_delta` :
   - `mode="min"` : amélioration si $\text{score} < \text{best} - \text{min\_delta}$ ;
   - `mode="max"` : amélioration si $\text{score} > \text{best} + \text{min\_delta}$.
3. **Si amélioration** : on mémorise le score, l'epoch (`best_epoch`), et — si
   `restore_best_weights` — une copie (`deepcopy`) de `model.state_dict()`. Le compteur
   d'attente est remis à zéro.
4. **Sinon** : le compteur d'attente est incrémenté. Quand il atteint `patience`,
   `step()` renvoie `True` et la boucle d'entraînement s'interrompt (`break`).
5. **En fin d'entraînement** : `early_stopping.restore(model)` recharge les poids de la
   meilleure epoch.

Paramètres :

| Paramètre | Rôle | Réglage typique |
|---|---|---|
| `monitor` | métrique surveillée | `val_loss` (plus stable que l'accuracy) |
| `mode` | sens de l'amélioration | `min` pour une loss, `max` pour une accuracy / un F1 |
| `patience` | epochs sans amélioration tolérées | 5–10 selon le bruit de la validation |
| `min_delta` | amélioration minimale pour « compter » | petit ; utile si la métrique bruite autour d'un palier |
| `restore_best_weights` | recharger le meilleur modèle à la fin | `True` (sinon on garde les poids déjà dégradés) |

### 3.3 Pièges

- **`patience` trop faible** : on s'arrête sur une simple fluctuation de la `val_loss`
  (fréquent avec nos classes rares) avant que le modèle n'ait convergé.
- **`patience` trop forte** : on gaspille du calcul à entraîner un modèle déjà en
  surapprentissage — mais `restore_best_weights` protège la **qualité** du résultat, le
  coût est seulement en temps.
- **Surveiller l'accuracy plutôt que la loss** : l'accuracy est en marches d'escalier
  et plus bruitée ; la `val_loss` capte plus tôt le début du surapprentissage.
- **Fuite de données** : le *split* de validation doit être stratifié et disjoint du
  test (déjà le cas via `stratify_fn` dans `BaseModel.fit`).

---

## 4. Interaction scheduler ↔ early stopping

Les deux mécanismes s'appuient sur le **même signal** (`val_loss`) mais avec des
objectifs différents : le scheduler **réagit** à la stagnation (réduire $\eta$),
l'early stopping **conclut** de la stagnation (arrêter).

### 4.1 Règle de patience

Si l'early stopping coupe **avant** que la réduction de $\eta$ ait eu le temps de
produire son effet, le scheduler ne sert à rien. Il faut donc :

$$\text{patience}_{\text{early stopping}} \ \gtrsim\ 2 \times \text{patience}_{\text{scheduler}}$$

Exemple cohérent : `--scheduler_patience 2 --early_stopping_patience 6`. Le scheduler
réduit $\eta$ après 2 epochs de stagnation ; l'early stopping laisse encore 4 epochs au
taux réduit pour prouver son utilité avant d'arrêter.

### 4.2 Horizon des schedulers aveugles

`cosine` et `step` utilisent `nb_epoch` comme horizon (`T_max`, `step_size`). Si
l'early stopping coupe à l'epoch 20 alors que `nb_epoch=50`, la fin du planning n'est
jamais atteinte — le cosinus s'arrête à mi-descente. Deux options :

- avec `cosine` / `step` : early stopping avec **grande `patience`** (garde-fou
  uniquement), ou pas d'early stopping du tout ;
- avec `plateau` : early stopping serré, les deux sont pilotés par les données et se
  complètent naturellement.

### 4.3 Garde-fou d'implémentation

`_train_and_validate` lève une `ValueError` si `early_stopping` **ou**
`ReduceLROnPlateau` est demandé **sans** jeu de validation
(`val_dataset` / `val_size`). Le même contrôle est fait en amont dans `src/train.py`
(`--validation no` incompatible avec `--scheduler plateau` et
`--early_stopping_patience`).

---

## 5. Implémentation dans le projet

| Élément | Emplacement | Rôle |
|---|---|---|
| `EarlyStopping` | `src/models/callbacks.py` | `step()` / `restore()`, expose `best_score`, `best_epoch`, `stopped_epoch` |
| `scheduler_factory=` | `BaseModel.fit` | callable `optimizer -> scheduler`, instancié après l'optimiseur |
| `early_stopping=` | `BaseModel.fit` | instance `EarlyStopping` ou `None` |
| boucle | `BaseModel._train_and_validate` | `scheduler.step()` + `early_stopping.step()` par epoch, `restore()` en fin |
| `history["lr"]` | `_train_and_validate` | taux d'apprentissage enregistré par epoch |
| `build_scheduler_factory` | `src/train.py` | construit la factory selon `--scheduler` |
| CLI | `src/train.py` | `--scheduler`, `--scheduler_patience`, `--lr_min`, `--early_stopping_patience` |
| courbe | `plot_train_metrics` (`src/models/helpers.py`) | `figures/training_lr.png` |

### 5.1 Comportement en validation croisée

En validation croisée (`cv` dans `BaseModel.fit`), chaque *fold* :

- **recrée** l'optimiseur puis le scheduler via la factory → pas de fuite d'état d'un
  *fold* à l'autre ;
- reçoit une **copie neuve** de l'`EarlyStopping` (`deepcopy`) → les compteurs et les
  poids mémorisés sont propres à chaque *fold*.

### 5.2 Exemples de commandes

```bash
# Exploration : plateau + early stopping, budget large
python -m src.train --scheduler plateau --scheduler_patience 2 \
    --early_stopping_patience 6 --nb_epoch 50

# Run final : décroissance cosinus sur budget fixé, sans early stopping serré
python -m src.train --scheduler cosine --lr_min 1e-5 --nb_epoch 40

# Baseline de comparaison : taux constant
python -m src.train --scheduler none --nb_epoch 5
```

---

## 6. Tableau récapitulatif : problème → mécanisme → solution

| Problème | Mécanisme | Solution retenue |
|---|---|---|
| $\eta$ fixe : trop grand en fin d'entraînement (oscillation), trop petit au début (lenteur) | Décroissance de $\eta$ au cours du temps | scheduler (`plateau` / `cosine` / `step`) |
| On ne sait pas *quand* réduire $\eta$ | Réduction déclenchée par la stagnation de `val_loss` | `ReduceLROnPlateau` (`--scheduler plateau`) |
| On veut une convergence fine sans paliers ni réglage fin | Décroissance cosinus vers $\eta_{\min}$ | `CosineAnnealingLR` (`--scheduler cosine`) |
| Reproduire une recette connue | Paliers fixes $\times\,\gamma$ | `StepLR` (`--scheduler step`) |
| Surapprentissage si trop d'epochs | Arrêt quand `val_loss` cesse de baisser | `EarlyStopping`, `patience` |
| Fixer `nb_epoch` à la main pour chaque config du grid search | Borne haute + arrêt automatique | `--early_stopping_patience`, `--nb_epoch` large |
| Garder par erreur le modèle sur-entraîné de la dernière epoch | Snapshot du meilleur `state_dict` | `restore_best_weights=True` |
| `val_loss` bruitée (classes rares) → faux plateaux / faux arrêts | Seuil d'amélioration + patience | `min_delta`, `patience` calibrée |
| Scheduler et early stopping qui se gênent | `patience_es` ≳ 2 × `patience_sched` | règle de réglage (§4.1) |
| État de scheduler / early stopping qui fuit entre *folds* | Recréation par *fold* | factory + `deepcopy` (§5.1) |

---

## 7. Recommandations pour le grid search

1. **Baseline** : `--scheduler none`, `nb_epoch` modéré — point de référence.
2. **Configuration par défaut du grid search** : `--scheduler plateau
   --scheduler_patience 2 --early_stopping_patience 6 --nb_epoch 50`. Robuste, ne
   demande aucune connaissance a priori, borne le coût.
3. **Meilleures configurations rejouées en « run final »** : `--scheduler cosine
   --lr_min 1e-5` avec `nb_epoch` calé sur l'epoch d'arrêt observée en phase 2, sans
   early stopping serré, pour maximiser le score final.
4. **Journaliser** `history["lr"]`, `best_epoch` et `stopped_epoch` avec les métriques
   (cf. `reports/todo.md`, point 4) pour comparer les configurations.

---

## 8. Bibliographie

- L. N. Smith, *Cyclical Learning Rates for Training Neural Networks*, WACV 2017.
- L. N. Smith, *A disciplined approach to neural network hyper-parameters: Part 1 --
  learning rate, batch size, momentum, and weight decay* (One-Cycle), 2018.
- I. Loshchilov, F. Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*,
  ICLR 2017 (cosine annealing).
- PyTorch, *torch.optim — How to adjust learning rate*,
  <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>.
- L. Prechelt, *Early Stopping -- But When?*, in *Neural Networks: Tricks of the
  Trade*, Springer, 1998.
- I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016 —
  chap. 7.8 (Early Stopping) et 8.3 (learning rate).
