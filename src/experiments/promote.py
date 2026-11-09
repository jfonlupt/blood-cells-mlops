"""Sélectionne les meilleurs runs existants et les réentraîne."""

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

from src.config import ExperimentConfig
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
            m[key] for m in metrics
            if isinstance(m.get(key), (int, float))
        ]
        if values:
            out[f"{key}_mean"] = statistics.fmean(values)
            out[f"{key}_std"] = (
                statistics.pstdev(values)
                if len(values) > 1 else 0.0
            )
    return out


def _load_config(run_dir: Path) -> ExperimentConfig:
    path = run_dir / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config.yaml introuvable : {path}")

    return ExperimentConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def _resolve_run_dir(value: str, root: Path) -> Path:
    """Résout run_dir sans créer results/results/..."""
    path = Path(value)

    if path.is_absolute():
        return path

    candidates = (
        path,
        root / path.name,
        root / path,
        root.parent / path,
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return root / path.name


def _load_checkpoint(path: Path, model: torch.nn.Module) -> None:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint.pt introuvable : {path}")

    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state = torch.load(path, map_location="cpu")

    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)


def _promoted_config(
    config: ExperimentConfig,
    additional_epochs: int,
) -> ExperimentConfig:
    """Même config que le parent, avec uniquement nb_epoch modifié."""
    data = config.model_dump(mode="json")
    data["train"]["nb_epoch"] += additional_epochs
    data["output"]["save_checkpoint"] = True
    return ExperimentConfig.model_validate(data)


def _merge_history(run_dir: Path, new: dict) -> dict:
    path = run_dir / "history.json"

    if not path.exists():
        return new

    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return new

    if not isinstance(old, dict):
        return new

    result = {}
    for key in set(old) | set(new):
        a, b = old.get(key, []), new.get(key, [])
        result[key] = a + b if isinstance(a, list) and isinstance(b, list) else b

    return result


def _train(
    parent_dir: Path,
    source: ExperimentConfig,
    config: ExperimentConfig,
    additional_epochs: int,
    root: Path,
) -> RunResult:
    started = time.time()
    seed_everything(source.seed)

    task = get_task(config.task)
    device = resolve_device(config.train.device)

    df = create_dataframe(config.data, task)
    _warn_classes(df, task)
    train_df, test_df = split_off_test(
        df, config.data, seed=config.seed
    )

    normalizer = (
        build_reinhard(
            train_df,
            config.transform,
            seed=config.seed,
        )
        if config.transform.reinhard else None
    )

    tb = TransformBuilder(
        config.transform,
        task,
        reinhard_normalizer=normalizer,
    )

    test_ds = DatasetTransform(
        BloodCellDataset(test_df, task),
        tb.build(train_df, training=False),
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
            "promote supporte uniquement le holdout : "
            "le checkpoint actuel ne contient que le premier fold."
        )

    tr_df, va_df = splits[0]

    model = build_model(
        config.model,
        task,
        img_size=config.transform.img_size,
    ).to(device)

    _load_checkpoint(
        parent_dir / "checkpoint.pt",
        model,
    )

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
        BloodCellDataset(tr_df, task),
        tb.build(tr_df, training=True),
    )

    val_ds = (
        DatasetTransform(
            BloodCellDataset(va_df, task),
            tb.build(tr_df, training=False),
        )
        if va_df is not None else None
    )

    # config.train.nb_epoch = nombre TOTAL après promotion.
    # Trainer ne doit faire que les nouvelles epochs.
    trainer_config = config.train.model_copy(
        update={"nb_epoch": additional_epochs}
    )

    trainer = Trainer(
        model,
        trainer_config,
        criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=build_metrics(task),
        callbacks=callbacks,
        device=device,
    )

    new_history = trainer.fit(train_ds, val_ds)
    history = _merge_history(parent_dir, new_history)

    scalars = trainer.evaluate(
        test_ds,
        metrics=build_metrics(
            task,
            for_test=True,
            device=device,
        ),
    )

    result = RunResult(
        config=config,
        task=task,
        histories=[history],
        test_metrics=_aggregate([
            {
                k: v for k, v in scalars.items()
                if isinstance(v, (int, float))
            }
        ]),
        test_metrics_folds=[{
            k: v for k, v in scalars.items()
            if isinstance(v, (int, float))
        }],
        best_state=model.state_dict(),
        duration_s=time.time() - started,
    )

    result.test_metrics_raw = scalars

    write_run_dir(result, root)
    append_leaderboard(root / "leaderboard.csv", result)

    return result


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

    required = {"run_dir", "status", "train.nb_epoch", metric}
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
        frame["status"].astype(str).str.lower().eq("ok")
        & pd.to_numeric(
            frame["train.nb_epoch"],
            errors="coerce",
        ).eq(source_epochs)
    ].copy()

    frame[metric] = pd.to_numeric(
        frame[metric],
        errors="coerce",
    )

    frame = frame.dropna(subset=[metric])

    if frame.empty:
        raise ValueError(
            f"Aucun run valide à {source_epochs} epochs."
        )

    frame = frame.sort_values(
        metric,
        ascending=not maximize,
    )

    n = max(
        1,
        int(np.ceil(len(frame) * top_percent / 100)),
    )

    return frame.head(n)


def promote(
    root: Path,
    source_epochs: int,
    additional_epochs: int,
    top_percent: float,
    metric: str = "val_f1_macro",
    maximize: bool = True,
    resume: bool = False,
) -> None:
    """Sélectionne puis entraîne les meilleurs modèles."""
    if source_epochs < 1:
        raise ValueError("source_epochs doit être >= 1.")
    if additional_epochs < 1:
        raise ValueError("additional_epochs doit être >= 1.")
    if not 0 < top_percent <= 100:
        raise ValueError("top_percent doit être entre 0 et 100.")

    selected = _select(
        root,
        source_epochs,
        top_percent,
        metric,
        maximize,
    )

    total = len(selected)
    target_epochs = source_epochs + additional_epochs

    print(
        f"Promotion : {total} modèles | "
        f"{source_epochs} → {target_epochs} epochs | "
        f"top {top_percent:g}% | {metric}"
    )

    for i, (_, row) in enumerate(
        selected.iterrows(),
        1,
    ):
        parent_dir = _resolve_run_dir(
            str(row["run_dir"]),
            root,
        )

        try:
            source_config = _load_config(parent_dir)

            if int(source_config.train.nb_epoch) != source_epochs:
                raise ValueError(
                    f"Configuration incohérente pour {parent_dir.name}."
                )

            checkpoint = parent_dir / "checkpoint.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint absent : {checkpoint}"
                )

            config = _promoted_config(
                source_config,
                additional_epochs,
            )

            target_dir = root / dirname_for(config)

            if resume and target_dir.exists():
                print(
                    f"[{i}/{total}] "
                    f"{parent_dir.name} → déjà fait "
                    f"({target_dir.name})"
                )
                continue

            print(
                f"[{i}/{total}] "
                f"{parent_dir.name} | "
                f"{metric}={float(row[metric]):.4f}"
            )

            result = _train(
                parent_dir,
                source_config,
                config,
                additional_epochs,
                root,
            )

            metadata = {
                "parent_run": str(parent_dir),
                "parent_run_name": parent_dir.name,
                "parent_epochs": source_epochs,
                "additional_epochs": additional_epochs,
                "total_epochs": target_epochs,
                "selection_metric": metric,
                "selection_score": float(row[metric]),
            }

            (result.run_dir / "promotion.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            print(
                f"       → {result.run_dir.name}"
            )

        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            print(f"       → ÉCHEC : {exc}")

    print("Promotion terminée.")
