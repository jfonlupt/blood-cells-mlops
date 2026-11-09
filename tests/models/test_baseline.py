"""Modèle baseline : forward inchangé après le refactor features/classifier."""

import torch

from src.models.baseline import BaselineClassifier


def test_forward_output_shape_unchanged():
    model = BaselineClassifier(num_classes=10)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)


def test_features_returns_flattened_vector_before_classifier():
    model = BaselineClassifier(num_classes=10)
    feats = model.features(torch.randn(2, 3, 224, 224))
    assert feats.shape == (2, 64 * 27 * 27)


def test_classifier_applied_to_features_matches_forward():
    model = BaselineClassifier(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        direct = model(x)
        via_features = model.classifier(model.features(x))
    assert torch.equal(direct, via_features)
