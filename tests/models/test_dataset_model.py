"""Modèle dataset (Barcelone vs Munich) : forward inchangé après le refactor features/classifier."""

import torch

from src.models.dataset_model import DatasetClassifier


def test_forward_output_shape_unchanged():
    model = DatasetClassifier(img_size=(224, 224))
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_features_returns_vector_before_classifier():
    model = DatasetClassifier(img_size=(224, 224))
    feats = model.features(torch.randn(2, 3, 224, 224))
    assert feats.ndim == 2
    assert feats.shape[0] == 2


def test_classifier_applied_to_features_matches_forward():
    model = DatasetClassifier(img_size=(224, 224))
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        direct = model(x)
        via_features = model.classifier(model.features(x))
    assert torch.equal(direct, via_features)
