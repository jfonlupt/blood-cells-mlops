"""Construction de la référence de drift (échantillonnage + descripteurs + embeddings)."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from src.experiments.drift_reference import (
    DriftReference,
    _pool_embedding,
    build_drift_reference,
    sample_stratified,
)


def _df(counts: dict) -> pd.DataFrame:
    rows = []
    for label, n in counts.items():
        rows += [{"label": label, "path": f"/fake/{label}_{i}.png"} for i in range(n)]
    return pd.DataFrame(rows)


def test_caps_at_n_per_class():
    df = _df({"a": 300, "b": 150})
    sample = sample_stratified(df, "label", n_per_class=200, seed=0)
    counts = sample["label"].value_counts()
    assert counts["a"] == 200
    assert counts["b"] == 150


def test_class_with_fewer_rows_than_requested_keeps_all():
    df = _df({"a": 5, "b": 300})
    sample = sample_stratified(df, "label", n_per_class=200, seed=0)
    assert (sample["label"] == "a").sum() == 5
    assert (sample["label"] == "b").sum() == 200


def test_deterministic_with_fixed_seed():
    df = _df({"a": 300})
    first = sample_stratified(df, "label", n_per_class=50, seed=42)
    second = sample_stratified(df, "label", n_per_class=50, seed=42)
    assert list(first["path"]) == list(second["path"])


def test_different_seeds_can_give_different_samples():
    df = _df({"a": 300})
    first = sample_stratified(df, "label", n_per_class=50, seed=1)
    second = sample_stratified(df, "label", n_per_class=50, seed=2)
    assert list(first["path"]) != list(second["path"])


def _image_df(tmp_path, n=6):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(np.uint8(rng.random((30, 30, 3)) * 255)).save(p)
        rows.append({"path": str(p), "label": i % 2})
    return pd.DataFrame(rows)


class _ModelWithFeatures(nn.Module):
    def __init__(self, out_dim=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim

    def features(self, x):
        pooled = self.pool(x).flatten(1)  # (B, 3)
        return pooled.repeat(1, self.out_dim // 3 + 1)[:, : self.out_dim]

    def forward(self, x):
        return self.features(x)


class _ModelWithoutFeatures(nn.Module):
    def forward(self, x):
        return x


class _ModelWithHighDimFeatures(nn.Module):
    """Imite BaselineClassifier.features() : un vecteur brut de grande dimension, sans
    pooling global, comme celui qui gonflait l'embedding de référence à ~90 Mo."""

    def __init__(self, out_dim=300):
        super().__init__()
        self.out_dim = out_dim

    def features(self, x):
        flat = x.flatten(1)  # (B, 3*H*W)
        return flat.repeat(1, self.out_dim // flat.shape[1] + 1)[:, : self.out_dim]

    def forward(self, x):
        return self.features(x)


def test_pool_embedding_is_noop_below_target_dim():
    features = torch.arange(12.0).reshape(2, 6)
    pooled = _pool_embedding(features, target_dim=128)
    assert torch.equal(pooled, features)


def test_pool_embedding_averages_contiguous_chunks():
    # 1 ligne, 6 colonnes -> 2 blocs de 3 : moyennes (0+1+2)/3=1 et (3+4+5)/3=4
    features = torch.arange(6.0).reshape(1, 6)
    pooled = _pool_embedding(features, target_dim=2)
    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled, torch.tensor([[1.0, 4.0]]))


def test_pool_embedding_gives_exactly_target_dim_even_when_not_evenly_divisible():
    features = torch.randn(3, 46656)
    pooled = _pool_embedding(features, target_dim=128)
    assert pooled.shape == (3, 128)


def test_build_drift_reference_caps_embedding_dimension_for_high_dim_model(tmp_path):
    """Régression : un modèle sans pooling global (ex. BaselineClassifier) ne doit plus
    produire un embedding.parquet de référence proportionnel à sa dimension brute."""
    df = _image_df(tmp_path)
    ref = build_drift_reference(
        df["path"].tolist(), _identity_transform, _ModelWithHighDimFeatures(out_dim=300), device="cpu"
    )
    emb_cols = [c for c in ref.embeddings.columns if c != "path"]
    assert len(emb_cols) == 128


def _identity_transform(image):
    return image.float() / 255.0


def test_build_drift_reference_includes_embeddings_when_model_supports_features(tmp_path):
    df = _image_df(tmp_path)
    ref = build_drift_reference(
        df["path"].tolist(), _identity_transform, _ModelWithFeatures(), device="cpu"
    )
    assert isinstance(ref, DriftReference)
    assert len(ref.descriptors) == 6
    assert ref.embeddings is not None
    assert len(ref.embeddings) == 6
    assert "path" in ref.embeddings.columns
    emb_cols = [c for c in ref.embeddings.columns if c != "path"]
    assert len(emb_cols) == 4


def test_build_drift_reference_merges_consistently_when_called_with_path_objects(tmp_path):
    """Reproduit le bug de prod : server.py /evaluate appelle build_drift_reference avec des
    pathlib.Path (chemins temporaires BentoML), pas des str. Si les deux DataFrames produits
    n'utilisent pas le même type de colonne "path", un merge("path") en aval (drift.py /
    server.py) ne matche plus aucune ligne -- symptôme observé : "empty column" levé par
    evidently sur un current_dataset vide après merge.
    """
    df = _image_df(tmp_path)
    path_objects = [Path(p) for p in df["path"]]

    ref = build_drift_reference(path_objects, _identity_transform, _ModelWithFeatures(), device="cpu")

    assert ref.descriptors["path"].map(type).eq(str).all()
    assert ref.embeddings["path"].map(type).eq(str).all()
    merged = ref.descriptors.merge(ref.embeddings, on="path")
    assert len(merged) == 6


def test_build_drift_reference_skips_embeddings_without_features(tmp_path):
    df = _image_df(tmp_path)
    ref = build_drift_reference(
        df["path"].tolist(), _identity_transform, _ModelWithoutFeatures(), device="cpu"
    )
    assert ref.embeddings is None
    assert len(ref.descriptors) == 6


def test_build_drift_reference_pooling_is_consistent_across_batch_sizes(tmp_path):
    """Le pooling moyenne par bloc de colonnes (dim=1), jamais entre lignes : le découper
    en plusieurs batches ne doit donc rien changer au résultat final par rapport à un
    seul batch couvrant tout l'échantillon."""
    df = _image_df(tmp_path)
    small_batches = build_drift_reference(
        df["path"].tolist(),
        _identity_transform,
        _ModelWithHighDimFeatures(out_dim=300),
        device="cpu",
        batch_size=2,
    )
    one_batch = build_drift_reference(
        df["path"].tolist(),
        _identity_transform,
        _ModelWithHighDimFeatures(out_dim=300),
        device="cpu",
        batch_size=32,
    )
    pd.testing.assert_frame_equal(small_batches.embeddings, one_batch.embeddings)


def test_build_drift_reference_batches_large_samples(tmp_path):
    df = _image_df(tmp_path, n=6)
    ref = build_drift_reference(
        df["path"].tolist(), _identity_transform, _ModelWithFeatures(), device="cpu", batch_size=2
    )
    assert len(ref.embeddings) == 6


def test_build_drift_reference_restores_training_mode_from_train(tmp_path):
    df = _image_df(tmp_path)
    model = _ModelWithFeatures()
    model.train()  # Explicitly set to training mode
    assert model.training is True
    build_drift_reference(df["path"].tolist(), _identity_transform, model, device="cpu")
    assert model.training is True


def test_build_drift_reference_restores_training_mode_from_eval(tmp_path):
    df = _image_df(tmp_path)
    model = _ModelWithFeatures()
    model.eval()  # Explicitly set to eval mode
    assert model.training is False
    build_drift_reference(df["path"].tolist(), _identity_transform, model, device="cpu")
    assert model.training is False


def test_build_drift_reference_handles_uneven_batch_division(tmp_path):
    df = _image_df(tmp_path, n=6)
    # batch_size=4 on n=6 gives: batch of 4, batch of 2 (uneven division)
    ref_uneven = build_drift_reference(
        df["path"].tolist(), _identity_transform, _ModelWithFeatures(), device="cpu", batch_size=4
    )
    # Compare to even batch division for row count and data integrity
    ref_even = build_drift_reference(
        df["path"].tolist(), _identity_transform, _ModelWithFeatures(), device="cpu", batch_size=32
    )
    assert len(ref_uneven.embeddings) == 6
    assert len(ref_even.embeddings) == 6
    assert not ref_uneven.embeddings.isna().any().any()
    assert not ref_even.embeddings.isna().any().any()
