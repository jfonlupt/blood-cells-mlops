"""Descripteurs visuels bas-niveau pour la référence de drift."""

import numpy as np
import pytest
from PIL import Image

from src.features.drift_descriptors import compute_descriptors


def _solid_image(path, color, size=(50, 40)):
    """Image unie de couleur ``color`` (R, G, B), taille ``(w, h)``."""
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = color
    Image.fromarray(arr).save(path)
    return str(path)


def test_uniform_red_image_has_expected_brightness_and_dims(tmp_path):
    path = _solid_image(tmp_path / "red.png", (255, 0, 0), size=(40, 40))
    df = compute_descriptors([path])
    row = df.iloc[0]
    assert row["width"] == 40
    assert row["height"] == 40
    assert row["aspect_ratio"] == pytest.approx(40 / 40)
    assert row["brightness"] == pytest.approx(1 / 3, abs=0.02)
    assert row["contrast"] == pytest.approx(0.0, abs=1e-6)  # image unie : pas de variation
    assert row["sharpness"] == pytest.approx(0.0, abs=1e-4)  # image unie : pas d'arête


def test_half_black_half_white_image_has_high_contrast(tmp_path):
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[20:, :, :] = 255
    path = tmp_path / "half.png"
    Image.fromarray(arr).save(path)
    df = compute_descriptors([str(path)])
    assert df.iloc[0]["contrast"] == pytest.approx(0.5, abs=0.02)  # écart-type d'un signal 0/1


def test_uniform_image_histogram_concentrated_in_expected_bins(tmp_path):
    path = _solid_image(tmp_path / "red.png", (255, 0, 0), size=(20, 20))
    df = compute_descriptors([path])
    row = df.iloc[0]
    assert row["hist_r_7"] == pytest.approx(1.0, abs=1e-6)  # canal R au max -> dernier bin
    assert row["hist_g_0"] == pytest.approx(1.0, abs=1e-6)  # canal G au min -> premier bin
    assert row["hist_b_0"] == pytest.approx(1.0, abs=1e-6)


def test_unreadable_image_raises_clear_error(tmp_path):
    bad = tmp_path / "not_an_image.png"
    bad.write_text("ceci n'est pas une image")
    with pytest.raises(OSError, match=str(bad)):
        compute_descriptors([str(bad)])


def test_output_has_one_row_per_image(tmp_path):
    paths = [_solid_image(tmp_path / f"img_{i}.png", (i * 10, 0, 0)) for i in range(5)]
    df = compute_descriptors(paths)
    assert len(df) == 5
    assert list(df["path"]) == paths
