"""Grad-CAM : sélection de couche, calcul de la carte, rendu des figures."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.models.gradcam import (
    GradCAM,
    default_gradcam_layer,
    last_conv_layer,
    last_spatial_layer,
    render,
    resolve_layer,
)


class _TinyCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def test_last_conv_layer_picks_the_last_conv():
    layer = last_conv_layer(_TinyCNN())
    assert isinstance(layer, nn.Conv2d)
    assert layer.out_channels == 8


def test_last_conv_layer_raises_without_conv():
    with pytest.raises(ValueError, match="Conv2d"):
        last_conv_layer(nn.Sequential(nn.Linear(4, 2)))


def test_resolve_layer_by_name_and_missing():
    model = _TinyCNN()
    assert resolve_layer(model, "features.2") is model.features[2]
    with pytest.raises(KeyError):
        resolve_layer(model, "does.not.exist")


def test_gradcam_compute_shape_and_range():
    torch.manual_seed(0)
    model = _TinyCNN().eval()
    cam = GradCAM(model, last_conv_layer(model)).compute(torch.randn(1, 3, 32, 32), target_index=1)
    assert cam.shape == (32, 32)
    assert cam.min() >= 0.0
    assert cam.max() == pytest.approx(1.0)  # normalisée min-max


def test_gradcam_target_index_changes_the_map():
    torch.manual_seed(0)
    model = _TinyCNN().eval()
    x = torch.randn(1, 3, 32, 32)
    cam_a = GradCAM(model, last_conv_layer(model)).compute(x, target_index=0)
    cam_b = GradCAM(model, last_conv_layer(model)).compute(x, target_index=2)
    assert not np.allclose(cam_a, cam_b)


def test_gradcam_defaults_to_predicted_class():
    torch.manual_seed(0)
    model = _TinyCNN().eval()
    x = torch.randn(1, 3, 32, 32)
    predicted = int(model(x).argmax())
    auto = GradCAM(model, last_conv_layer(model)).compute(x, target_index=None)
    explicit = GradCAM(model, last_conv_layer(model)).compute(x, target_index=predicted)
    assert np.allclose(auto, explicit)


class _SEishCNN(nn.Module):
    """CNN avec une branche squeeze-excite : la dernière Conv2d sort en 1x1 (piège)."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.stem = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Conv2d(6, 6, 1)  # 1x1 -> (B, 6, 1, 1)
        self.fc = nn.Linear(6, n_classes)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        s = torch.sigmoid(self.se(self.pool(x)))
        return self.fc((x * s).mean(dim=(2, 3)))


def test_last_spatial_layer_skips_1x1_squeeze_excite():
    model = _SEishCNN()
    layer = last_spatial_layer(model, probe_shape=(1, 3, 16, 16))
    assert layer is model.stem


def test_last_spatial_layer_raises_without_spatial_output():
    with pytest.raises(ValueError, match="spatial"):
        last_spatial_layer(nn.Sequential(nn.Linear(4, 2)), probe_shape=(1, 4))


def test_default_gradcam_layer_falls_back_to_last_conv_for_plain_cnn():
    model = _TinyCNN()
    assert default_gradcam_layer(model) is last_conv_layer(model)


def test_default_gradcam_layer_targets_backbone_for_transfer_model():
    from src.config import TransferConfig
    from src.models.transfer import TransferModel

    torch.manual_seed(0)
    model = TransferModel(TransferConfig(backbone="resnet18", pretrained=False), num_classes=3).eval()
    layer = default_gradcam_layer(model)
    assert isinstance(layer, nn.Conv2d)
    assert layer is model.backbone.layer4[1].conv2  # dernière conv du dernier étage

    # le hook arrière se déclenche (couche feuille) : la carte a la bonne forme et est bornée
    cam = GradCAM(model, layer).compute(torch.randn(1, 3, 224, 224), target_index=1)
    assert cam.shape == (224, 224)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0


def test_gradcam_works_on_fully_frozen_transfer_backbone():
    from src.config import TransferConfig
    from src.models.transfer import TransferModel

    torch.manual_seed(0)
    model = TransferModel(
        TransferConfig(backbone="resnet18", pretrained=False, freeze="all"), num_classes=3
    ).eval()
    layer = default_gradcam_layer(model)
    cam = GradCAM(model, layer).compute(torch.randn(1, 3, 224, 224), target_index=0)
    assert cam.shape == (224, 224)  # le hook arrière s'est déclenché malgré le backbone gelé


def test_render_creates_dir_and_four_images(tmp_path):
    rng = np.random.default_rng(0)
    img = tmp_path / "cell.png"
    Image.fromarray((rng.random((20, 30, 3)) * 255).astype(np.uint8)).save(img)
    cam = rng.random((64, 64)).astype(np.float32)
    out_dir = tmp_path / "gradcam" / "run"  # n'existe pas encore

    paths = render(str(img), cam, out_dir)

    assert set(paths) == {"original", "heatmap", "overlay", "panel"}
    for name in ("original", "heatmap", "overlay"):
        assert (out_dir / f"cell_{name}.png").exists()
    assert (out_dir / "cell_gradcam.png").exists()  # figure combinée + colorbar
    assert paths["panel"] == out_dir / "cell_gradcam.png"
