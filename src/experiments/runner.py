"""Exécution d'une expérience unique : config -> entraînement -> artefacts."""

from __future__ import annotations

import random
import statistics
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from coolname import generate_slug

from src.experiments import tracking
from src.experiments.artifacts import RunResult, append_leaderboard, write_run_dir
from src.experiments.drift_reference import build_drift_reference, sample_stratified
from src.features.dataframe import create_dataframe
from src.features.dataset import BloodCellDataset, DatasetTransform
from src.features.stats import build_reinhard
from src.features.transform import build_transform
from src.models import build_model
from src.models.losses import build_loss
from src.models.metrics import build_metrics
from src.tasks import get_task
from src.training.callbacks import EarlyStopping
from src.training.components import (
    TransformBuilder,
    build_optimizer,
    build_sampler,
    build_scheduler,
    make_splits,
    optimizer_param_groups,
    resolve_device,
    split_off_test,
)
from src.training.trainer import Trainer

if TYPE_CHECKING:
    import pandas as pd

    from src.config import ExperimentConfig
    from src.tasks import TaskSpec


def seed_everything(seed: int) -> None:
    """Fixe les graines de ``random``, ``numpy`` et ``torch`` pour la reproductibilité.

    Args:
        seed: Graine à propager à tous les générateurs pseudo-aléatoires.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- graine du RNG legacy utilisé par sklearn
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warn_if_missing_classes(df: pd.DataFrame, task: TaskSpec) -> None:
    """Émet un avertissement si des classes de la tâche sont absentes du DataFrame.

    Args:
        df: DataFrame échantillonné pour l'expérience.
        task: Spécification de la tâche (colonne de label et nombre de classes).
    """
    present = df[task.label_column].nunique()
    if present < task.num_classes:
        warnings.warn(
            f"{task.name}: {present}/{task.num_classes} classes présentes dans le DataFrame.",
            stacklevel=2,
        )


def _best_fold_index(histories: list[dict]) -> int:
    """Index du fold dont le ``val_loss`` minimal (sur ses epochs) est le plus bas.

    En holdout/``validation="none"``, un seul fold : renvoie toujours 0. En
    cross-validation, compare les folds entre eux sur ce même critère que
    ``_val_f1_macro`` (meilleure epoch = ``val_loss`` minimal), pour éviter de
    garder arbitrairement le modèle du premier fold entraîné (cf.
    ``reports/rapport_difficultes_pieges.md``). Si aucun fold n'a de
    ``val_loss`` (``train.validation="none"``), renvoie 0.

    Args:
        histories: Historique d'entraînement par fold (sortie de ``trainer.fit``).

    Returns:
        L'index, dans ``histories`` (et donc dans la liste ``models`` parallèle),
        du fold retenu.
    """
    scores = [min(h["val_loss"]) for h in histories if h.get("val_loss")]
    if len(scores) != len(histories):
        return 0
    return min(range(len(scores)), key=lambda i: scores[i])


def _aggregate(fold_metrics: list[dict]) -> dict:
    """Agrège les métriques par fold en un dict unique.

    Args:
        fold_metrics: Liste des dicts de métriques scalaires, un par fold.

    Returns:
        Le dict du fold unique en holdout ; sinon ``{"<clé>_mean": ..., "<clé>_std": ...}``
        pour chaque métrique scalaire.
    """
    if not fold_metrics:
        return {}
    if len(fold_metrics) == 1:
        return dict(fold_metrics[0])
    keys = fold_metrics[0]
    out: dict = {}
    for key in keys:
        values = [m[key] for m in fold_metrics if isinstance(m.get(key), (int, float))]
        if values:
            out[f"{key}_mean"] = statistics.fmean(values)
            out[f"{key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return out


def _preprocessing_stats(transform_builder, reinhard_normalizer) -> dict:
    """Statistiques de prétraitement à persister (``preprocessing.json``).

    12 flottants suffisant à reconstruire le transform d'évaluation sans les
    données : moyenne / écart-type du dataset (par canal RGB) et, si Reinhard
    est actif, les stats cible Lab du normaliseur.
    """
    return {
        "dataset_mean": transform_builder.dataset_mean.tolist(),
        "dataset_std": transform_builder.dataset_std.tolist(),
        "reinhard": (
            {
                "target_means": reinhard_normalizer.target_means.tolist(),
                "target_stds": reinhard_normalizer.target_stds.tolist(),
            }
            if reinhard_normalizer is not None
            else None
        ),
    }


def run_experiment(
    config: ExperimentConfig,
    output_root: Path | None = None,
    name: str | None = None,
    evaluate_test: bool = True,
) -> RunResult:
    """Lance une expérience, écrit son dossier + une ligne de leaderboard.

    Args:
        config: Configuration complète de l'expérience.
        output_root: Racine des artefacts ; par défaut ``config.output.root``.
        name: Nom lisible du run (colonne ``name`` du leaderboard, ``meta.json``).
            Si ``None``, un nom aléatoire du type ``brave-otter`` est tiré. Ce nom
            n'entre pas dans l'identité du run (hash de config).
        evaluate_test: Si faux, saute l'évaluation sur ``test_ds`` (``test_metrics``
            reste vide). À utiliser pour toute comparaison entre plusieurs
            configurations (grid search) : le jeu de test ne doit être évalué
            qu'une seule fois, sur la configuration finale retenue après
            sélection sur validation — sinon le test set fuite dans la
            sélection de modèle (cf. ``reports/rapport_difficultes_pieges.md``).

    Returns:
        Le ``RunResult`` complet, avec ``run_dir`` renseigné.
    """
    started = time.time()
    name = name or generate_slug(2)
    root = Path(output_root) if output_root is not None else config.output.root
    seed_everything(config.seed)
    task = get_task(config.task)
    device = resolve_device(config.train.device)
    if config.verbose:
        print(f"Device utilisé : {device}")

    df = create_dataframe(config.data, task)
    _warn_if_missing_classes(df, task)
    train_df, test_df = split_off_test(df, config.data, seed=config.seed)

    normalizer = (
        build_reinhard(train_df, config.transform, seed=config.seed)
        if config.transform.reinhard
        else None
    )
    tb = TransformBuilder(config.transform, task, reinhard_normalizer=normalizer)
    # tb.build() fait un vrai passage de sonde (compute_dataset_mean_std, ~640 images
    # pretraitees) : on ne le paie que si test_ds sert reellement (evaluate_test=True).
    test_ds = (
        DatasetTransform(BloodCellDataset(test_df, task), tb.build(train_df, training=False))
        if evaluate_test
        else None
    )

    histories: list[dict] = []
    fold_test_scalar: list[dict] = []
    fold_test_raw: list[dict] = []
    fold_val_scalar: list[dict] = []
    fold_val_raw: list[dict] = []
    models: list = []
    interrupted = False

    for tr_df, va_df in make_splits(train_df, config.train, config.data, seed=config.seed):
        model = build_model(config.model, task, img_size=config.transform.img_size).to(device)
        criterion = build_loss(config.loss, task, tr_df).to(device)
        optimizer = build_optimizer(config.optim, optimizer_param_groups(model, config.optim.lr))
        scheduler = build_scheduler(config.scheduler, optimizer, nb_epoch=config.train.nb_epoch)
        callbacks: list = []
        if config.train.early_stopping_patience:
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=config.train.early_stopping_patience,
                )
            ]

        sampler = build_sampler(config.train, tr_df[task.label_column].to_numpy(), task.num_classes)

        train_ds = DatasetTransform(BloodCellDataset(tr_df, task), tb.build(tr_df, training=True))
        val_ds = (
            DatasetTransform(BloodCellDataset(va_df, task), tb.build(tr_df, training=False))
            if va_df is not None
            else None
        )

        trainer = Trainer(
            model,
            config.train,
            criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=build_metrics(task),
            callbacks=callbacks,
            device=device,
            train_sampler=sampler,
        )
        try:
            histories.append(trainer.fit(train_ds, val_ds))
        except KeyboardInterrupt:
            histories.append(dict(trainer.history))
            interrupted = True

        if evaluate_test:
            scalars = trainer.evaluate(
                test_ds, metrics=build_metrics(task, for_test=True, device=device)
            )
            # ``Trainer.evaluate`` clone la collection en interne et renvoie déjà les
            # métriques préfixées ``test_`` (scalaires ``.item()`` ou vecteurs ``.tolist()``).
            fold_test_scalar.append(
                {k: v for k, v in scalars.items() if isinstance(v, (int, float))}
            )
            fold_test_raw.append(dict(scalars))

        if val_ds is not None:
            # Métriques riches (f1 macro, mcc, per-class, min_critical_recall...) sur
            # validation : jamais sur test_ds, donc utilisable pour comparer les
            # combinaisons d'un grid search sans fuite (cf.
            # reports/rapport_difficultes_pieges.md).
            val_scalars = trainer.evaluate(
                val_ds, metrics=build_metrics(task, for_test=True, device=device), prefix="val"
            )
            fold_val_scalar.append(
                {k: v for k, v in val_scalars.items() if isinstance(v, (int, float))}
            )
            fold_val_raw.append(dict(val_scalars))
        models.append(model)

        if interrupted:
            print("Run interrompu : sauvegarde de l'état atteint avant l'arrêt.")
            break

    duration = time.time() - started
    if config.verbose:
        print(f"Model trained in {duration:.2f}s")
    best_model = models[_best_fold_index(histories)]
    result = RunResult(
        config=config,
        task=task,
        histories=histories,
        test_metrics=_aggregate(fold_test_scalar),
        test_metrics_folds=fold_test_scalar,
        best_state=best_model.state_dict(),
        duration_s=duration,
        name=name,
        status="interrupted" if interrupted else "ok",
        n_trainable_params=sum(p.numel() for p in best_model.parameters() if p.requires_grad),
    )
    result.test_metrics_raw = fold_test_raw[0] if fold_test_raw else None
    result.val_metrics = _aggregate(fold_val_scalar)
    result.val_metrics_raw = fold_val_raw[0] if fold_val_raw else None
    result.preprocessing = _preprocessing_stats(tb, normalizer)

    eval_transform = build_transform(
        config.transform,
        training=False,
        reinhard_normalizer=normalizer,
        dataset_mean=tb.dataset_mean,
        dataset_std=tb.dataset_std,
    )
    drift_sample = sample_stratified(train_df, task.label_column, n_per_class=200, seed=config.seed)
    result.drift_reference = build_drift_reference(
        drift_sample["path"].tolist(), eval_transform, best_model, device=device
    )

    write_run_dir(result, root, config.verbose)
    append_leaderboard(root / "leaderboard.csv", result)
    x = (
        torch.stack([train_ds[i][0] for i in range(2)]).cpu().numpy()
        if config.output.save_checkpoint
        else None
    )
    tracking.log_run(result, result.run_dir, best_model.cpu(), x)
    if interrupted:
        # Etat sauvegardé : on relève pour stopper le script appelant (ex. un
        # grid search ne doit pas enchaîner sur la combinaison suivante).
        raise KeyboardInterrupt
    return result
