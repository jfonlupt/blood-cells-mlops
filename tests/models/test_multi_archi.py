"""Modèle multi-blocs : forward inchangé après le refactor features/classifier."""

import torch

from src.models.multi_archi_model import MultiArchiModel


def _block(depth=1, channels=8, kernel=3):
    from types import SimpleNamespace

    return SimpleNamespace(depth=depth, channels=channels, kernel=kernel)


def test_forward_output_shape_with_gap():
    model = MultiArchiModel(
        num_classes=10, blocks=[_block()], classifier=[16], img_size=(32, 32), use_gap=True
    )
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_forward_output_shape_without_gap():
    model = MultiArchiModel(
        num_classes=10, blocks=[_block()], classifier=[16], img_size=(32, 32), use_gap=False
    )
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_features_returns_vector_before_classifier():
    model = MultiArchiModel(
        num_classes=10, blocks=[_block()], classifier=[16], img_size=(32, 32), use_gap=True
    )
    feats = model.features(torch.randn(2, 3, 32, 32))
    assert feats.shape == (2, 8)  # use_gap=True -> (B, out_channels du dernier bloc)


def test_classifier_applied_to_features_matches_forward():
    model = MultiArchiModel(
        num_classes=10, blocks=[_block()], classifier=[16], img_size=(32, 32), use_gap=True
    )
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        direct = model(x)
        via_features = model.classifier(model.features(x))
    assert torch.equal(direct, via_features)
