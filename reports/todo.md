# Liste des TODO de Romain

1. ~~Faire des plots de la loss et des métriques d'entrainement~~
2. ~~Ajouter la loss ponderée~~
5. ~~Ajouter learning rate scheduler et early stop~~
3. ~~Ajouter le WeightedRandomSampler~~ (`train.sampler: weight` + `train.sampler_alpha` : exposant sur 1/fréquence, tunable pour coupler avec la loss pondérée)
4. ~~Sauvegarder le modèle, les graphs et les métriques pour les meilleurs modèles~~
6. ~~Créer un Grid Search~~ (pipeline `src.experiments.grid_cli` + grilles `configs/grids/` livrés)
    - ~~baseline (no transformation, no augmentation, cross entropy simple)~~
    - ~~transformation : reinhard, clahe, jpeg~~ (`configs/grids/preprocessing.yaml`)
    - ~~augmentation : rotation, flip, color jitter avec des paramètres différents~~ (`configs/grids/augmentation.yaml`)
    - deséquilibre : ~~cross entropy simple, focal simple pour différent beta, WeightedRandomSampler seul et couplé à la focal~~ (`configs/grids/imbalance.yaml`)
7. ~~Test sur type, criticality, criticality binaire~~ (option `--task {type,criticality,binary_criticality}` de la CLI)
8. ~~Ajout courbe AU PRC~~ (`pr_curve` dans `build_metrics(for_test=True)` ; figure `metrics_<split>_pr_curve.png` par classe avec AP annotée, sauvée avec les autres figures du run)
9. ~~Ajout d'un nom au runs~~ (`--name` de `run` / `grid_cli`, sinon nom aléatoire `coolname` ; colonne `name` du leaderboard + `meta.json`. Runs déplacés dans `results/runs/`, leaderboard reste `results/leaderboard.csv`)
10. ~~Ajout d'un runner d'inférence~~ (`src.experiments.infer --run <run_dir> --image …` ; recharge config+checkpoint, reconstruit le prétraitement via `preprocessing.json` — mean/std + stats Reinhard sauvés par le runner — ou rejeu depuis le seed pour les runs antérieurs)
11. ~~Ajout d'un runner pour grad-cam~~ (`src.experiments.gradcam --run <run_dir> --image … --out … [--layer] [--target-class]` ; Grad-CAM maison dans `src/models/gradcam.py`, sortie : original / heatmap / overlay + figure combinée avec colorbar)
12. ~~Métriques top-3~~ (`top3_acc`, `top3_f1`, `top3_recall_per_class` dans `build_metrics(for_test=True)`, seulement si `num_classes ≥ 5`)
13. changer dans le rapport, cnn binaire dataset, le 0.5 est du à un effondrement du modèle, pas aux transformations
