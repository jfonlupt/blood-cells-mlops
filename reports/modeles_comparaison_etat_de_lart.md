---
title: "Modèles de comparaison — état de l'art pour la classification de cellules sanguines"
author: "Romain"
date: "8 septembre 2026"
---

# Modèles de comparaison — état de l'art

**Ébauche de rapport — projet Blood Cell Classification**

> Document de travail. Il recense les modèles issus de la littérature qui
> traitent la même tâche que ce projet (classification de cellules sanguines
> isolées), en vue de les implémenter et de les entraîner sur nos données pour
> les comparer au CNN *baseline* (`src/models/baseline.py`). Il alimente les
> sections encore vides du rapport principal : « Modèle avancé »,
> « Transfer Learning » et « Modèle hiérarchique » (`reports/report.tex`).

---

## 1. Rappel de la tâche

- **Entrée** : image de cellule sanguine isolée (frottis de sang périphérique),
  prétraitée en `3 × 224 × 224`.
- **Cibles** (sélectionnées par `task` dans la config) :
  - `type` — 10 classes fusionnées (Basophile, Blaste, Éosinophile,
    Érythroblaste, Granulocytes immatures, Lymphocyte, Monocyte, Neutrophile,
    Plaquette, Promyélocyte) ;
  - `criticality` — 3 paliers (`normal` / `intermediaire` / `critique`) ;
  - `binary_criticality` — 2 classes (`normal` / `abnormal`) ;
  - `dataset` — 2 classes (`barcelona` / `munich`), tâche de contrôle.
- **Données** : fusion **Acevedo-20 (PBC, Barcelone)** + **Matek-19 (LMU,
  Munich)**, 35 457 images.
- **Difficultés** : déséquilibre long-tail (rapport ~17:1 après fusion),
  **domain gap** marqué entre les deux sites — un CNN retrouve la source quasi
  parfaitement quel que soit le prétraitement.
- **Témoin actuel** : petit CNN maison, ~3,0 M paramètres, dont 99 % dans la
  première couche dense (pas de *global pooling*, pas de *batch-norm*).

Les métriques de comparaison sont celles définies au §Métriques du rapport
principal : **MCC** et **F1 macro** pour la vue d'ensemble, **rappel des classes
critiques** (blaste, promyélocyte) comme grandeur décisive, **AUPRC** par classe,
et sous-matrice de confusion « anormal → normal ».

---

## 2. Modèles de référence de la littérature (mêmes jeux de données)

Ces travaux utilisent **Acevedo-20 et/ou Matek-19**, ce qui en fait des points
de comparaison directs.

| Travail | Modèle | Intérêt pour nous |
|---|---|---|
| **Matek et al. 2019**, *Nature Machine Intelligence* [1] (déjà cité) | **ResNeXt-50** (pré-entraîné ImageNet) | Référence historique sur LMU/Matek ; ~90 % accuracy, performance « human-level » sur la question binaire blaste / non-blaste. C'est le benchmark clinique sur les blastes. |
| **Acevedo et al. 2019**, *Comput. Methods Programs Biomed.* [2] (déjà cité) | **VGG-16** + **InceptionV3** (fine-tuning) | Référence historique sur PBC ; ~96 %. |
| Littérature PBC 2023–2025 [5,6] | **EfficientNet-B0/B3**, **EfficientNetV2-S**, **DenseNet-121** | 98–98,6 % sur PBC seul (8 classes). SOTA « CNN classique » sur ce type d'images. |
| **Salehi et al. 2022** (marrlab), *Unsupervised Cross-Domain Feature Extraction for Single Blood Cell Image Classification* [3] | Auto-encodeur cross-domain entraîné sur **Matek + Acevedo** | Traite explicitement le *domain shift* entre nos deux sources. |
| **DaNet** 2025 [7] ; *Continual learning for cross-domain WBC* 2023 [8] | Alignement de features / apprentissage continu, protocole *leave-one-domain-out* sur Matek / Acevedo / PBC | Protocole d'évaluation cross-site à reprendre pour lire nos métriques mono-site. |

---

## 3. Backbones *transfer learning* à implémenter

C'est le cœur du travail de comparaison. Tous chargeables via **`timm`**
(`timm.create_model(name, pretrained=True)`), avec **global average pooling +
tête** ajoutée par-dessus — ce qui corrige la faiblesse structurelle de la
*baseline* (paramètres concentrés dans le *flatten* → dense, sensibilité à la
position de la cellule). Implémentation : §7.

| Modèle `timm` | Params | Pourquoi le tester |
|---|---|---|
| `mobilenetv3_large_100` **ou** `efficientnet_b0` | 4–5 M | Léger, quasi-SOTA sur PBC, meilleur rapport performance / coût de calcul. |
| `resnet50` | 25 M | Standard universel, comparable à Matek (ResNeXt-50). |
| `densenet121` | 8 M | Très utilisé sur PBC, robuste en petit régime de données. |
| `efficientnetv2_s` | 21 M | Souvent le meilleur CNN sur ce type de tâche. |
| `convnext_tiny` | 28 M | CNN moderne, forte régularisation. |
| `vit_base_patch16_224` / `deit3_small_patch16_224` / `swin_tiny_patch4_window7_224` | 22–86 M | Vision Transformers : jugés compétitifs dans la littérature WBC [9] mais gourmands en données / augmentation — intéressant à confronter à notre déséquilibre. |

**Priorité recommandée** : commencer par `efficientnet_b0`, `resnet50`,
`densenet121` (couvre l'essentiel de la littérature), puis ajouter
`convnext_tiny` et un ViT si le budget de calcul le permet.

**Variantes d'entraînement à balayer** pour chaque backbone :

- *feature extractor* figé (backbone gelé) vs fine-tuning complet vs fine-tuning
  progressif (*gradual unfreezing*) ;
- combinaison avec les stratégies de rééquilibrage déjà en place (sampler
  `alpha`, `class-balanced focal loss`, boost clinique `extra_weight`) —
  cf. grille `configs/grids/imbalance.yaml`.

---

## 4. Modèle de fondation spécifique à l'hématologie — DinoBloom

**DinoBloom** (marrlab, MICCAI 2024) [4].
GitHub : <https://github.com/marrlab/DinoBloom> ·
Poids : <https://zenodo.org/records/10908163> · arXiv : `2404.05022`.

- ViT entraîné en self-supervised (pipeline **DINOv2**) sur **13 jeux de données
  d'hématologie = plus de 380 000 images de leucocytes**, **incluant Matek-19**.
  **Acevedo-20 est en revanche explicitement exclu du pré-entraînement** par les
  auteurs, qui le gardent comme jeu de test externe : *« We train the DinoBloom
  models on all datasets except the Acevedo dataset, which is kept as external
  test set »* [4].
- Variants : **DinoBloom-S** (embeddings 384-d, 22 M params) →
  **DinoBloom-B** (768-d, 86 M) → **-L** (1024-d, 304 M) → **-G** (1536-d, 1,1 B).
  Entrée ~`224 × 224`.
- Objectif affiché : **généraliser malgré les *batch effects*** entre
  institutions — précisément notre problème « le CNN retrouve toujours la
  source ». Annoncé comme battant les modèles vision médicaux et généralistes
  sur la classification de type cellulaire et le sous-typage LMA.
- **Deux usages à comparer** :
  1. **extracteur figé** + tête linéaire ou MLP (baseline rapide, peu de calcul) ;
  2. **fine-tuning** léger du ViT.

**Réserve méthodologique** : DinoBloom a *vu* Matek-19 pendant son pré-entraînement
self-supervised (sans les labels), mais **pas** Acevedo-20, gardé par ses auteurs
comme jeu de test externe. Ce n'est donc pas un test de généralisation
« zéro-shot » pur sur l'ensemble de notre fusion — mais ce n'est pas non plus une
contamination totale : côté DinoBloom, Acevedo reste un domaine non vu.

Reste que, côté **nôtre**, Acevedo fait partie des données servant à entraîner
notre CNN *baseline* et nos autres backbones ; ce n'est donc pas un jeu « propre »
au sens où on l'entend habituellement (jamais utilisé par personne dans la
chaîne). Et chercher un dataset WBC public totalement absent du corpus DinoBloom
est un problème plus large que le nôtre : un travail récent qui a cherché
exactement ça pour auditer DinoBloom sous *acquisition shift* a dû conclure que
tous les grands jeux WBC publics usuels (Matek, Raabin, MLL23/LabAnonymous) sont
déjà dans ses 13 datasets — *« there is no leakage-free DinoBloom target »* [11].

**À présenter dans le rapport** :
1. Formulation principale (recommandée) : la comparaison DinoBloom vs ImageNet
   est un test de *transfert vers un domaine partiellement vu* — pas un
   zéro-shot — en citant [11] pour montrer qu'il s'agit d'une limite connue et
   documentée de la littérature actuelle, pas d'un oubli méthodologique.
2. Expérience annexe possible, à présenter séparément du chiffre principal :
   entraîner la tête de classification uniquement sur Matek et évaluer le
   DinoBloom gelé sur Acevedo seul (non vu par lui) pour isoler la contribution
   du *domain gap* — non comparable telle quelle au pipeline fusionné.

---

## 5. Attaquer le *domain gap* directement (piste notée « non testée » dans le rapport)

Le rapport principal mentionne l'entraînement adversarial de domaine sans l'avoir
mis en œuvre. Le câblage existe déjà : une tâche `dataset` et un classifieur de
source.

- **DANN / *gradient reversal*** (Ganin & Lempitsky 2016) : une couche de
  renversement de gradient + un classifieur de source branché sur les features,
  entraîné pour que le réseau *ne puisse pas* prédire le site. Implémentation
  ~30 lignes par-dessus n'importe quel backbone `timm`. C'est un **complément**
  aux modèles du §3, pas un concurrent : on l'active ou non sur un même backbone.
- Alternative plus simple déjà évoquée dans le rapport : *fine-tuning* séparé par
  site puis moyennage / ensembling.

Métrique de succès : faire **baisser l'accuracy de la tâche `dataset`** vers 0,5
*sans* dégrader le F1 macro de la tâche `type`, et surtout stabiliser les
métriques des classes mono-site (plaquette, blaste).

---

## 6. Modèle hiérarchique (section vide du rapport)

Approche *two-stage* cohérente avec la criticité clinique :

1. **Étage 1** — détecteur binaire `normal` / `abnormal` réglé à **haut rappel**
   sur `abnormal` (seuil ajusté en post-traitement, cf. `select_best_checkpoint`).
2. **Étage 2** — classifieur fin (10 classes, ou seulement les classes anormales)
   appliqué aux cellules jugées anormales par l'étage 1.

Réutilise deux backbones du §3. À comparer au modèle plat 10 classes sur :
rappel critique, nombre d'erreurs « anormal → normal », coût de calcul à
l'inférence.

Variante : hiérarchie guidée par la taxonomie (lignée myéloïde / lymphoïde /
non-cellule), à rapprocher du *decoupling representation / classifier*
(Kang et al. 2020, déjà dans `references.bib`).

---

## 7. Intégration dans le pipeline (implémentée)

Le pipeline est piloté par `model.name`. Le type **`transfer`** est en place
(librairie : **timm**, fine-tuning **niveau 2** : gel + tête MLP + dropout + LR
discriminatif).

### Config

`model.transfer` (`src/config.py::TransferConfig`) :

| Champ | Valeurs | Effet |
|---|---|---|
| `backbone` | tout nom timm | ex. `resnet50`, `efficientnet_b0`, `vit_base_patch16_224.augreg2_in21k_ft_in1k` |
| `pretrained` | `true` / `false` | `false` = même archi, poids aléatoires (run de contrôle) |
| `freeze` | `none` / `all` / entier `N` | `all` = *linear probing* (backbone gelé, BN/LN forcées en `eval`) ; `N` = seuls les `N` derniers groupes de stages du backbone restent entraînables (via `backbone.group_matcher` : resnet50 → 5 groupes, efficientnet_b0 → 8, vit_base → 14), le reste gelé mais BN non figée |
| `head_hidden` | `[]`, `[256]`, … | couches cachées du MLP après *global average pooling* ; `[]` = `Linear` direct |
| `head_dropout` | `0.0`–`1.0` | dropout dans la tête |
| `backbone_lr_mult` | `≥ 0` | lr backbone = `optim.lr × mult` ; `0` ≈ gelé, `1` = fine-tuning naïf. S'applique aux groupes non gelés (utile combiné à `freeze=N`) |

`transform.norm_stats` (`dataset` par défaut) : `imagenet` utilise les constantes
ImageNet attendues par les poids pré-entraînés et **court-circuite la sonde**
`compute_dataset_mean_std`. `configs/transfer.yaml` la met à `imagenet`. Avec
`norm_stats="dataset"`, `build_transform` **exige** désormais `dataset_mean` /
`dataset_std` (plus de défaut silencieux).

### Fichiers

- `src/models/transfer.py` — `TransferModel` : `timm.create_model(num_classes=0,
  global_pool="avg")` + tête `_build_head`, gel (`all` / `_freeze_early_groups(N)`),
  `train()` surchargé (backbone en `eval` si `freeze="all"`),
  `param_groups(base_lr)` (LR discriminatif).
- `src/models/__init__.py` — branche `"transfer"` de `build_model`.
- `src/training/components.py` — `optimizer_param_groups(model, base_lr)` :
  délègue à `model.param_groups` si présent, sinon `model.parameters()`.
- `src/experiments/runner.py` — l'optimiseur reçoit ces groupes.
- `src/features/transform.py` — `IMAGENET_MEAN/STD`, `norm_stats`, `dataset_mean/std`
  requis pour `norm_stats="dataset"`.
- `src/features/stats.py` — la sonde `compute_dataset_mean_std` force
  `norm_stats="dataset"` (mesure les stats brutes quel que soit le réglage d'entraînement).
- `configs/transfer.yaml` — config de base.
- `configs/grids/architecture/transfer.yaml` — grille des 7 backbones du §3
  (linear probing + fine-tuning × 3 `backbone_lr_mult` = 28 combinaisons).

### Lancer

```bash
uv run python -m src.experiments.grid_cli \
  --config configs/transfer.yaml --grid configs/grids/architecture/transfer.yaml
```

Combinable avec les grilles `imbalance` / `preprocessing` (axes orthogonaux).
Premier lancement : téléchargement des poids depuis le HF Hub (cache ensuite).
Sur MPS, garder `PYTORCH_ENABLE_MPS_FALLBACK=1` ; batch 64 en 224 px tient sur
16 Go, descendre à 32 sur 8 Go.

### Grad-CAM

`src/models/gradcam.py::default_gradcam_layer` couvre les trois familles :
`baseline` / `multi_archi` → dernière `Conv2d` ; `transfer` → dernière conv
spatiale du backbone via `last_spatial_layer` (passe *sonde*, ignore les convs
1×1 des blocs *squeeze-excite*, choisit une feuille pour que le hook arrière se
déclenche). `GradCAM.compute` force `requires_grad` sur l'entrée pour fonctionner
même avec un backbone entièrement gelé. **ViT / Swin non couverts** (les
activations ne sont pas des cartes conv) — limiter l'analyse Grad-CAM aux CNN, ou
adopter `pytorch-grad-cam` pour les transformers.

### Reste à faire

- **DinoBloom** : ajouter `model.transfer.source: timm | dinobloom` (défaut
  `timm`) ; la branche `dinobloom` construit un ViT DINOv2 et y charge les poids
  Zenodo. Non implémenté (hook prévu, cf. §4).
- **DANN** : flag `model.domain_adv` + couche *gradient reversal* + head de
  source, réutilisant les labels `dataset`. Non implémenté (cf. §5).

---

## Bibliographie

[1] Matek, C., Schwarz, S., Spiekermann, K., Marr, C. *Human-level recognition of
blast cells in acute myeloid leukaemia with convolutional neural networks.*
Nature Machine Intelligence 1, 538–544 (2019).
<https://www.nature.com/articles/s42256-019-0101-9>

[2] Acevedo, A., Alférez, S., Merino, A., Puigví, L., Rodellar, J. *Recognition
of peripheral blood cell images using convolutional neural networks.* Comput.
Methods Programs Biomed. 180, 105020 (2019).

[3] Salehi, R. et al. *Unsupervised Cross-Domain Feature Extraction for Single
Blood Cell Image Classification.* MICCAI 2022.
<https://arxiv.org/pdf/2207.00501>

[4] Koch, V. et al. *DinoBloom: A Foundation Model for Generalizable Cell
Embeddings in Hematology.* MICCAI 2024.
<https://arxiv.org/abs/2404.05022> · <https://github.com/marrlab/DinoBloom>

[5] *Automatic classification of 10 blood cell subtypes using transfer learning
via pre-trained convolutional neural networks.* ScienceDirect (2024).
<https://www.sciencedirect.com/science/article/pii/S2352914824000984>

[6] *Classification of All Blood Cell Images using ML and DL Models.* arXiv
2308.06300. <https://arxiv.org/pdf/2308.06300>

[7] *DaNet: Domain-adaptive white blood cell classification through synthetic
augmentation and cross-domain feature alignment.* ScienceDirect (2025).
<https://www.sciencedirect.com/science/article/pii/S2590005625000438>

[8] *A Continual Learning Approach for Cross-Domain White Blood Cell
Classification.* arXiv 2308.12679. <https://arxiv.org/html/2308.12679>

[9] *White Blood Cell Classification: CNN and Vision Transformer under Medical
Microscope.* Algorithms 16(11), 525. <https://doi.org/10.3390/a16110525>

[10] Ganin, Y., Lempitsky, V. *Unsupervised Domain Adaptation by
Backpropagation.* ICML 2015 / *Domain-Adversarial Training of Neural Networks*,
JMLR 2016. <https://arxiv.org/abs/1505.07818>

[11] *Can You Trust Frozen Hematology Foundation Models under Acquisition
Shift?* (2026). Audite DinoBloom sous *acquisition/domain shift* ; constate
qu'aucun des grands jeux WBC publics usuels (Matek, Raabin, MLL23/LabAnonymous)
n'est exempt de fuite vis-à-vis du pré-entraînement DinoBloom — seul Acevedo est
tenu à l'écart par ses auteurs. <https://arxiv.org/abs/2608.25148>
