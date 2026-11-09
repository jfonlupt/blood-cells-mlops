"""Descripteurs visuels bas-niveau pour la référence de drift (sans Reinhard/CLAHE/Normalize).

Calculés sur un transform minimal (resize + mise à l'échelle [0,1]) pour ne
pas masquer un drift de coloration/scanner par la normalisation colorimétrique
appliquée au reste du pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.nn import functional as F

from src.constants import IMG_SIZE
from src.features.transform import ToUnitRange, load_image_rgb, resize_with_pad

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_N_HIST_BINS = 8
_LAPLACIAN_KERNEL = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _sharpness(gray: torch.Tensor) -> float:
    """Variance du Laplacien (netteté) sur une image en niveaux de gris, ``(H, W)`` dans ``[0, 1]``."""
    kernel = _LAPLACIAN_KERNEL.view(1, 1, 3, 3)
    # padding "replicate" (pas zero) : sinon même une image unie a un Laplacien non nul au bord
    padded = F.pad(gray.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    laplacian = F.conv2d(padded, kernel, padding=0)
    return laplacian.var().item()


def _color_histogram(image_01: torch.Tensor, n_bins: int = _N_HIST_BINS) -> dict:
    """Histogramme couleur normalisé par canal (``image_01`` : CHW float ``[0, 1]``)."""
    out = {}
    for channel, name in enumerate("rgb"):
        hist = torch.histc(image_01[channel], bins=n_bins, min=0.0, max=1.0)
        hist = hist / hist.sum().clamp(min=1.0)
        for i, value in enumerate(hist.tolist()):
            out[f"hist_{name}_{i}"] = value
    return out


def compute_descriptors(
    image_paths: Sequence[str | Path], img_size: tuple[int, int] = IMG_SIZE
) -> pd.DataFrame:
    """Descripteurs visuels bas-niveau, un par image de ``image_paths``.

    Args:
        image_paths: Chemins des images à décrire.
        img_size: Taille ``(H, W)`` à laquelle les images sont redimensionnées
            avant le calcul de la luminosité/contraste/histogramme/netteté
            (les dimensions d'origine sont, elles, gardées telles quelles
            dans ``width``/``height``).

    Returns:
        Un DataFrame avec une colonne ``path`` + les descripteurs numériques.

    Raises:
        OSError: si une image ne peut pas être lue, avec son chemin dans le message.
    """
    rows = []
    for path in image_paths:
        try:
            raw = load_image_rgb(path)  # CHW uint8
        except Exception as exc:
            raise OSError(f"image illisible : {path}") from exc
        _, orig_h, orig_w = raw.shape
        resized = resize_with_pad(raw, img_size, fill=255)
        image_01 = ToUnitRange()(resized)
        gray = image_01.mean(dim=0)
        row = {
            "path": str(path),
            "width": orig_w,
            "height": orig_h,
            "aspect_ratio": orig_w / orig_h,
            "brightness": image_01.mean().item(),
            "contrast": gray.std().item(),
            "sharpness": _sharpness(gray),
        }
        row.update(_color_histogram(image_01))
        rows.append(row)
    return pd.DataFrame(rows)
