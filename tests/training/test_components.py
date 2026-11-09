"""Tests des composants d'entraînement (splits, device, optim, scheduler, transforms)."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import WeightedRandomSampler

from src.config import DataConfig, OptimConfig, SchedulerConfig, TrainConfig, TransformConfig
from src.tasks import get_task
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


def _df(n=40):
    return pd.DataFrame(
        {"label": ([0, 1] * (n // 2)), "type_dataset": (["a", "b"] * (n // 2)), "dataset": "barcelona"}
    )


def test_resolve_device_returns_torch_device():
    assert isinstance(resolve_device("cpu"), torch.device)
    assert isinstance(resolve_device("auto"), torch.device)


def test_split_off_test_ratio():
    tr, te = split_off_test(_df(100), DataConfig(test_size=0.2, stratify="type_dataset"))
    assert len(te) == 20
    assert len(tr) == 80


def test_make_splits_holdout_yields_one_pair():
    splits = list(make_splits(_df(), TrainConfig(validation="holdout", val_size=0.25), DataConfig()))
    assert len(splits) == 1
    _tr, va = splits[0]
    assert va is not None
    assert len(va) == 10


def test_make_splits_none_yields_no_validation():
    splits = list(make_splits(_df(), TrainConfig(validation="none"), DataConfig()))
    assert len(splits) == 1
    assert splits[0][1] is None


def test_make_splits_cross_yields_k_folds():
    splits = list(
        make_splits(_df(60), TrainConfig(validation="cross", cross_val_folds=3), DataConfig())
    )
    assert len(splits) == 3
    for tr, va in splits:
        assert va is not None and len(tr) + len(va) == 60


def test_build_optimizer_dispatch():
    params = [torch.nn.Parameter(torch.randn(2))]
    assert build_optimizer(OptimConfig(name="adam", lr=1e-3), params).defaults["lr"] == 1e-3
    assert type(build_optimizer(OptimConfig(name="sgd"), params)).__name__ == "SGD"


def test_optimizer_param_groups_plain_model_returns_parameters():
    model = torch.nn.Linear(3, 2)
    params = list(optimizer_param_groups(model, base_lr=1e-3))
    assert params and all(isinstance(p, torch.nn.Parameter) for p in params)


def test_optimizer_param_groups_delegates_to_model_when_available():
    from src.config import TransferConfig
    from src.models.transfer import TransferModel

    model = TransferModel(
        TransferConfig(backbone="resnet18", pretrained=False, backbone_lr_mult=0.1), num_classes=3
    )
    groups = optimizer_param_groups(model, base_lr=1e-3)
    assert [g["lr"] for g in groups] == pytest.approx([1e-4, 1e-3])
    # utilisable directement par l'optimiseur
    opt = build_optimizer(OptimConfig(name="adam", lr=1e-3), groups)
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([1e-4, 1e-3])


def test_build_scheduler_dispatch():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.randn(2))], lr=0.1)
    assert build_scheduler(SchedulerConfig(name="none"), opt, nb_epoch=5) is None
    assert isinstance(build_scheduler(SchedulerConfig(name="plateau"), opt, nb_epoch=5), ReduceLROnPlateau)
    assert isinstance(build_scheduler(SchedulerConfig(name="cosine"), opt, nb_epoch=5), CosineAnnealingLR)
    assert isinstance(build_scheduler(SchedulerConfig(name="step"), opt, nb_epoch=6), StepLR)


_IMBALANCED = [0] * 10 + [1] * 30  # effectifs 10 / 30, ratio 3:1


def test_build_sampler_none_returns_none():
    assert build_sampler(TrainConfig(sampler="none"), _IMBALANCED, num_classes=2) is None


def test_build_sampler_weight_returns_weighted_sampler_with_replacement():
    s = build_sampler(TrainConfig(sampler="weight"), _IMBALANCED, num_classes=2)
    assert isinstance(s, WeightedRandomSampler)
    assert s.num_samples == 40
    assert s.replacement is True
    assert len(s.weights) == 40


def test_build_sampler_alpha_zero_is_uniform():
    s = build_sampler(TrainConfig(sampler="weight", sampler_alpha=0.0), _IMBALANCED, num_classes=2)
    assert torch.allclose(s.weights, s.weights[0].expand_as(s.weights))


def test_build_sampler_alpha_one_is_inverse_frequency():
    s = build_sampler(TrainConfig(sampler="weight", sampler_alpha=1.0), _IMBALANCED, num_classes=2)
    # poids(classe 0) / poids(classe 1) == effectif(classe 1) / effectif(classe 0)
    assert (s.weights[0] / s.weights[-1]).item() == pytest.approx(3.0)


def test_build_sampler_alpha_half_partially_rebalances():
    s = build_sampler(TrainConfig(sampler="weight", sampler_alpha=0.5), _IMBALANCED, num_classes=2)
    assert (s.weights[0] / s.weights[-1]).item() == pytest.approx(3.0**0.5)


def test_build_sampler_handles_class_absent_from_split():
    s = build_sampler(TrainConfig(sampler="weight"), [0, 0, 1], num_classes=3)
    assert torch.isfinite(s.weights).all()


def test_transform_builder_produces_compose(tmp_path):
    rng = np.random.default_rng()
    rows = []
    for i in range(6):
        p = tmp_path / f"i{i}.png"
        Image.fromarray((rng.random((24, 24, 3)) * 255).astype(np.uint8)).save(p)
        rows.append({"path": str(p), "label": i % 2, "dataset": "barcelona"})
    df = pd.DataFrame(rows)
    tb = TransformBuilder(TransformConfig(reinhard=False, clahe=False, jpeg=False), get_task("type"))
    compose = tb.build(df, training=False)
    out = compose(torch.randint(0, 255, (3, 24, 24), dtype=torch.uint8))
    assert out.shape == (3, 224, 224)


def test_transform_builder_imagenet_norm_skips_probe(tmp_path):
    from src.features.transform import IMAGENET_MEAN, IMAGENET_STD

    df = pd.DataFrame([{"path": "/does/not/exist.png", "label": 0, "dataset": "barcelona"}])
    tb = TransformBuilder(TransformConfig(norm_stats="imagenet"), get_task("type"))
    # aucun fichier lu : la sonde compute_dataset_mean_std est court-circuitée
    tb.build(df, training=False)
    assert tb.dataset_mean.tolist() == pytest.approx(list(IMAGENET_MEAN))
    assert tb.dataset_std.tolist() == pytest.approx(list(IMAGENET_STD))


def test_transform_builder_exposes_last_mean_std(tmp_path):
    rng = np.random.default_rng()
    rows = []
    for i in range(6):
        p = tmp_path / f"i{i}.png"
        Image.fromarray((rng.random((24, 24, 3)) * 255).astype(np.uint8)).save(p)
        rows.append({"path": str(p), "label": i % 2, "dataset": "barcelona"})
    df = pd.DataFrame(rows)
    tb = TransformBuilder(TransformConfig(reinhard=False, clahe=False, jpeg=False), get_task("type"))
    tb.build(df, training=False)
    assert tb.dataset_mean.shape == (3,)
    assert tb.dataset_std.shape == (3,)
