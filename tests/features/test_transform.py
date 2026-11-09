"""Tests de la fabrique de transforms."""

import pytest
import torch

from src.config import TransformConfig
from src.features.transform import build_transform

# Les tests qui ne portent pas sur la normalisation utilisent norm_stats="imagenet"
# (chemin sans stats de dataset) pour ne pas avoir à fournir mean/std.
_NO_COLOR = {"reinhard": False, "clahe": False, "norm_stats": "imagenet"}


def test_train_transform_has_augmentation_steps():
    t = build_transform(TransformConfig(**_NO_COLOR), training=True)
    kinds = [type(s).__name__ for s in t.transforms]
    assert kinds.count("RandomApply") >= 1
    assert "RandomHorizontalFlip" in kinds


def test_eval_transform_has_no_random_flip():
    t = build_transform(TransformConfig(**_NO_COLOR), training=False)
    kinds = [type(s).__name__ for s in t.transforms]
    assert "RandomHorizontalFlip" not in kinds
    assert kinds[-1] == "Normalize"


def test_clahe_toggle():
    with_clahe = build_transform(
        TransformConfig(reinhard=False, clahe=True, norm_stats="imagenet"), training=False
    )
    without = build_transform(TransformConfig(**_NO_COLOR), training=False)
    names = lambda c: [type(s).__name__ for s in c.transforms]
    assert "CLAHETransform" in names(with_clahe)
    assert "CLAHETransform" not in names(without)


def _normalize_step(compose):
    return next(s for s in compose.transforms if type(s).__name__ == "Normalize")


def test_norm_stats_dataset_uses_passed_mean_std():
    cfg = TransformConfig(reinhard=False, clahe=False)  # norm_stats defaults to "dataset"
    t = build_transform(
        cfg, training=False, dataset_mean=torch.tensor([0.1, 0.2, 0.3]),
        dataset_std=torch.tensor([0.4, 0.5, 0.6]),
    )
    step = _normalize_step(t)
    assert list(step.mean) == pytest.approx([0.1, 0.2, 0.3])
    assert list(step.std) == pytest.approx([0.4, 0.5, 0.6])


def test_norm_stats_imagenet_overrides_passed_mean_std():
    cfg = TransformConfig(reinhard=False, clahe=False, norm_stats="imagenet")
    t = build_transform(
        cfg, training=False, dataset_mean=torch.tensor([0.1, 0.2, 0.3]),
        dataset_std=torch.tensor([0.4, 0.5, 0.6]),
    )
    step = _normalize_step(t)
    assert list(step.mean) == pytest.approx([0.485, 0.456, 0.406])
    assert list(step.std) == pytest.approx([0.229, 0.224, 0.225])


def test_norm_stats_dataset_without_stats_raises():
    cfg = TransformConfig(reinhard=False, clahe=False)  # norm_stats="dataset"
    with pytest.raises(ValueError, match="dataset_mean"):
        build_transform(cfg, training=False)


def test_transform_runs_on_a_tensor():
    cfg = TransformConfig(reinhard=False, clahe=False, jpeg=False, norm_stats="imagenet")
    t = build_transform(cfg, training=True)
    out = t(torch.randint(0, 255, (3, 40, 40), dtype=torch.uint8))
    assert out.shape == (3, 224, 224)


def _find_random_gaussian_blur(compose):
    for step in compose.transforms:
        if type(step).__name__ == "RandomApply":
            for inner in step.transforms:
                if type(inner).__name__ == "GaussianBlur":
                    return step, inner
    return None


def test_train_transform_includes_gaussian_blur_with_configured_params():
    cfg = TransformConfig(
        reinhard=False, clahe=False, norm_stats="imagenet", p_blur=0.5, blur_kernel_size=3,
        blur_sigma_min=0.2, blur_sigma_max=1.5,
    )
    t = build_transform(cfg, training=True)
    found = _find_random_gaussian_blur(t)
    assert found is not None
    random_apply, blur = found
    assert random_apply.p == 0.5
    assert blur.kernel_size == (3, 3)
    assert blur.sigma == [0.2, 1.5]


def test_blur_not_present_in_eval():
    cfg = TransformConfig(reinhard=False, clahe=False, norm_stats="imagenet", p_blur=0.5)
    t = build_transform(cfg, training=False)
    assert _find_random_gaussian_blur(t) is None
