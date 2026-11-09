"""Réentraîne depuis zéro les meilleurs modèles avec une configuration modifiée."""

from __future__ import annotations

import json
import random
import statistics
import time
import traceback
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import yaml

from src.config import ExperimentConfig, deep_merge
from src.experiments.artifacts import (
    RunResult,
    append_leaderboard,
    dirname_for,
    write_run_dir,
)
from src.features.dataframe import create_dataframe
from src.features.dataset import BloodCellDataset, DatasetTransform
from src.features.stats import build_reinhard
from src.models import build_model
from src.models.losses import build_loss
from src.models.metrics import build_metrics
from src.tasks import get_task
from src.training.callbacks import EarlyStopping
from src.training.components import (
    TransformBuilder,
    build_optimizer,
    build_scheduler,
    make_splits,
    resolve_device,
    split_off_test,
)
from src.training.trainer import Trainer

if TYPE_CHECKING:
    from src.tasks import TaskSpec


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warn_classes(df: pd.DataFrame, task: TaskSpec) -> None:
    if df[task.label_column].nunique() < task.num_classes:
        warnings.warn(
            f"{task.name}: certaines classes sont absentes.",
            stacklevel=2,
        )


def _aggregate(metrics: list[dict]) -> dict:
    if len(metrics) == 1:
        return dict(metrics[0])

    out = {}
    for key in metrics[0]:
        values = [
            m[key]
            for m in metrics
            if isinstance(m.get(key), (int, float))
        ]
        if values:
            out[f"{key}_mean"] = statistics.fmean(values)
            out[f"{key}_std"] = (
                statistics.pstdev(values)
                if len(values) > 1
                else 0.0
            )
    return out


def _load_config(run_dir: Path) -> ExperimentConfig:
    path = run_dir / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config.yaml introuvable : {path}")

    return ExperimentConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def _load_retrain_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration retrain introuvable : {path}"
        )

    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(
            f"La configuration retrain doit être un dictionnaire : {path}"
        )

    return data


def _nest(key: str, value) -> dict:
    parts = key.split(".")
    result = value

    for part in reversed(parts):
        result = {part: result}

    return result


def _apply_sets(
    config: dict,
    overrides: list[str],
) -> dict:
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Override invalide : '{override}'. "
                "Format attendu : cle.valeur=valeur"
            )

        key, raw_value = override.split("=", 1)

        if not key:
            raise ValueError(
                f"Clé vide dans l'override : '{override}'"
            )

        value = yaml.safe_load(raw_value)
        config = deep_merge(
            config,
            _nest(key, value),
        )

    return config


def _build_config(
    parent_config: ExperimentConfig,
    retrain_file: Path,
    overrides: list[str],
) -> ExperimentConfig:
    """Parent config < retrain.yaml < --set."""
    config = parent_config.model_dump(mode="json")

    config = deep_merge(
        config,
        _load_retrain_config(retrain_file),
    )

    config = _apply_sets(
        config,
        overrides,
    )

    return ExperimentConfig.model_validate(config)


def _resolve_run_dir(
    value: str,
    root: Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    for candidate in (
        path,
        root / path.name,
        root / path,
        root.parent / path,
    ):
        if candidate.exists():
            return candidate

    return root / path.name


def _select(
    root: Path,
    source_epochs: int,
    top_percent: float,
    metric: str,
    maximize: bool,
) -> pd.DataFrame:
    path = root / "leaderboard.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"leaderboard.csv introuvable : {path}"
        )

    frame = pd.read_csv(path)

    required = {
        "run_dir",
        "status",
        "train.nb_epoch",
        metric,
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "Colonnes absentes du leaderboard : "
            + ", ".join(sorted(missing))
        )

    frame = frame.drop_duplicates(
        "run_dir",
        keep="last",
    )

    frame = frame[
        frame["status"]
        .astype(str)
        .str.lower()
        .eq("ok")
    ]

    epochs = pd.to_numeric(
        frame["train.nb_epoch"],
        errors="coerce",
    )

    frame = frame[
        epochs == source_epochs
    ].copy()

    frame[metric] = pd.to_numeric(
        frame[metric],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[metric]
    )

    if frame.empty:
        raise ValueError(
            f"Aucun run valide à {source_epochs} epochs "
            f"avec la métrique '{metric}'."
        )

    frame = frame.sort_values(
        metric,
        ascending=not maximize,
    )

    number_selected = max(
        1,
        int(
            np.ceil(
                len(frame)
                * top_percent
                / 100
            )
        ),
    )

    return frame.head(
        number_selected
    )


def _train(
    config: ExperimentConfig,
    root: Path,
) -> RunResult:
    started = time.time()

    seed_everything(config.seed)

    task = get_task(config.task)
    device = resolve_device(config.train.device)

    df = create_dataframe(
        config.data,
        task,
    )

    _warn_classes(
        df,
        task,
    )

    train_df, test_df = split_off_test(
        df,
        config.data,
        seed=config.seed,
    )

    normalizer = (
        build_reinhard(
            train_df,
            config.transform,
            seed=config.seed,
        )
        if config.transform.reinhard
        else None
    )

    tb = TransformBuilder(
        config.transform,
        task,
        reinhard_normalizer=normalizer,
    )

    test_ds = DatasetTransform(
        BloodCellDataset(
            test_df,
            task,
        ),
        tb.build(
            train_df,
            training=False,
        ),
    )

    splits = list(
        make_splits(
            train_df,
            config.train,
            config.data,
            seed=config.seed,
        )
    )

    if len(splits) != 1:
        raise ValueError(
            "retrain supporte uniquement le holdout."
        )

    tr_df, va_df = splits[0]

    # ---------------------------------------------------------------
    # NOUVEAU MODÈLE : AUCUN CHECKPOINT N'EST CHARGÉ.
    # ---------------------------------------------------------------
    model = build_model(
        config.model,
        task,
        img_size=config.transform.img_size,
    ).to(device)

    criterion = build_loss(
        config.loss,
        task,
        tr_df,
    ).to(device)

    optimizer = build_optimizer(
        config.optim,
        model.parameters(),
    )

    scheduler = build_scheduler(
        config.scheduler,
        optimizer,
        nb_epoch=config.train.nb_epoch,
    )

    callbacks = []

    if config.train.early_stopping_patience:
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config.train.early_stopping_patience,
            )
        ]

    train_ds = DatasetTransform(
        BloodCellDataset(
            tr_df,
            task,
        ),
        tb.build(
            tr_df,
            training=True,
        ),
    )

    val_ds = (
        DatasetTransform(
            BloodCellDataset(
                va_df,
                task,
            ),
            tb.build(
                tr_df,
                training=False,
            ),
        )
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
    )

    history = trainer.fit(
        train_ds,
        val_ds,
    )

    scalars = trainer.evaluate(
        test_ds,
        metrics=build_metrics(
            task,
            for_test=True,
            device=device,
        ),
    )

    scalar_metrics = {
        key: value
        for key, value in scalars.items()
        if isinstance(value, (int, float))
    }

    result = RunResult(
        config=config,
        task=task,
        histories=[history],
        test_metrics=_aggregate(
            [scalar_metrics]
        ),
        test_metrics_folds=[
            scalar_metrics
        ],
        best_state=model.state_dict(),
        duration_s=time.time() - started,
    )

    result.test_metrics_raw = scalars

    write_run_dir(
        result,
        root,
    )

    append_leaderboard(
        root / "leaderboard.csv",
        result,
    )

    return result


def retrain(
    root: Path,
    retrain_file: Path,
    source_epochs: int,
    additional_top_percent: float,
    overrides: list[str],
    metric: str = "val_f1_macro",
    maximize: bool = True,
    resume: bool = False,
) -> None:
    if source_epochs < 1:
        raise ValueError(
            "--from-epochs doit être >= 1."
        )

    if not 0 < additional_top_percent <= 100:
        raise ValueError(
            "--top doit être > 0 et <= 100."
        )

    root = Path(root)

    selected = _select(
        root=root,
        source_epochs=source_epochs,
        top_percent=additional_top_percent,
        metric=metric,
        maximize=maximize,
    )

    print()
    print("=" * 72)
    print("RETRAIN DEPUIS ZÉRO")
    print("=" * 72)
    print(f"Modèles sources : {len(selected)}")
    print(f"Source epochs   : {source_epochs}")
    print(f"Top             : {additional_top_percent:g}%")
    print(f"Métrique        : {metric}")
    print(f"Configuration   : {retrain_file}")
    print("Poids du parent : NON")
    print("=" * 72)

    for i, (_, row) in enumerate(
        selected.iterrows(),
        1,
    ):
        parent_dir = _resolve_run_dir(
            str(row["run_dir"]),
            root,
        )

        try:
            if not parent_dir.exists():
                raise FileNotFoundError(
                    f"Run introuvable : {parent_dir}"
                )

            parent_config = _load_config(
                parent_dir
            )

            if int(parent_config.train.nb_epoch) != source_epochs:
                raise ValueError(
                    f"{parent_dir.name}: "
                    f"configuration à "
                    f"{parent_config.train.nb_epoch} epochs."
                )

            config = _build_config(
                parent_config,
                retrain_file,
                overrides,
            )

            target_dir = (
                root
                / dirname_for(config)
            )

            if resume and target_dir.exists():
                print(
                    f"[{i}/{len(selected)}] "
                    f"{parent_dir.name} → "
                    f"déjà fait : {target_dir.name}"
                )
                continue

            print(
                f"[{i}/{len(selected)}] "
                f"{parent_dir.name} | "
                f"{metric}={float(row[metric]):.4f} "
                f"→ {config.train.nb_epoch} epochs"
            )

            result = _train(
                config,
                root,
            )

            if result.run_dir is None:
                raise RuntimeError(
                    "run_dir absent après entraînement."
                )

            metadata = {
                "type": "retrain",
                "parent_run": str(parent_dir),
                "parent_run_name": parent_dir.name,
                "parent_epochs": source_epochs,
                "training_from_scratch": True,
                "selection_metric": metric,
                "selection_score": float(row[metric]),
                "retrain_config": str(retrain_file),
                "overrides": overrides,
                "total_epochs": config.train.nb_epoch,
            }

            (result.run_dir / "retrain.json").write_text(
                json.dumps(
                    metadata,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"       → {result.run_dir.name}"
            )

        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(
                f"       → ÉCHEC : {exc}"
            )

    print()
    print("Retrain terminé.")
