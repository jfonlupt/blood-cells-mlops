"""Grad-CAM (Selvaraju et al. 2017) : carte de saillance d'un CNN sur une image.

Implémentation directe : hooks sur une couche de convolution pour récupérer ses
activations et le gradient du score cible, pondération canal par canal par la
moyenne spatiale du gradient, ReLU, ré-échantillonnage à la taille d'entrée,
normalisation min-max.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from torch import nn
from torch.nn import functional as F

from src.features.transform import load_image_rgb, resize_with_pad

if TYPE_CHECKING:
    import numpy.typing as npt


def last_conv_layer(model: nn.Module) -> nn.Module:
    """Renvoie la dernière ``nn.Conv2d`` rencontrée dans ``model`` (ordre des modules)."""
    conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv = module
    if conv is None:
        raise ValueError("aucune couche Conv2d dans le modèle — préciser --layer")
    return conv


def last_spatial_layer(
    model: nn.Module, probe_shape: tuple[int, ...] = (1, 3, 224, 224)
) -> nn.Module:
    """Dernier sous-module dont la sortie est une carte spatiale ``(B, C, H, W)`` avec ``H, W >= 2``.

    Repose sur une passe *sonde* (entrée de zéros, ``eval``, ``no_grad``) qui
    enregistre la forme de sortie de chaque module feuille. Contrairement à
    :func:`last_conv_layer`, ignore les convolutions ``1x1`` des blocs
    *squeeze-excite* (sortie ``(B, C, 1, 1)``) et fonctionne pour les backbones
    dont la dernière feuille utile n'est pas une ``Conv2d`` (BatchNorm, activation).

    Args:
        model: Module à sonder (typiquement un backbone timm).
        probe_shape: Forme du tenseur d'entrée de la sonde.

    Returns:
        Le sous-module cible.

    Raises:
        ValueError: si aucun module ne produit de sortie spatiale.
    """
    shapes: dict[str, torch.Size | None] = {}
    handles = []
    for name, module in model.named_modules():
        if not name or next(module.children(), None) is not None:
            continue  # on ne garde que les feuilles nommées

        def _record(_m, _i, output, _n=name) -> None:
            shapes[_n] = getattr(output, "shape", None)

        handles.append(module.register_forward_hook(_record))

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(probe_shape))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    modules = dict(model.named_modules())
    spatial = [
        name
        for name, shape in shapes.items()
        if shape is not None and len(shape) == 4 and shape[2] >= 2 and shape[3] >= 2
    ]
    if not spatial:
        raise ValueError("aucune couche à sortie spatiale — préciser --layer")

    # On préfère une conv (cible Grad-CAM canonique), puis une couche de
    # normalisation ; en dernier recours n'importe quelle carte spatiale. Cela
    # évite de tomber sur une activation ReLU *inplace*, incompatible avec le
    # hook arrière de Grad-CAM.
    convs = [n for n in spatial if isinstance(modules[n], nn.Conv2d)]
    norms = [n for n in spatial if "norm" in type(modules[n]).__name__.lower()]
    return modules[(convs or norms or spatial)[-1]]


def default_gradcam_layer(
    model: nn.Module, probe_shape: tuple[int, ...] = (1, 3, 224, 224)
) -> nn.Module:
    """Couche cible Grad-CAM par défaut.

    - modèle avec un backbone timm (``model.backbone`` exposant ``forward_features``,
      cas de ``TransferModel``) : dernière carte spatiale du backbone via
      :func:`last_spatial_layer` (dernière conv d'un étage : ``layer4.1.conv2``,
      ``conv_head``, ``stages.3.blocks.2.conv_dw``…) — un module *feuille*, pour
      que le hook arrière de Grad-CAM se déclenche (les conteneurs ``nn.Sequential``
      ne le déclenchent pas) ;
    - CNN « maison » (baseline, multi_archi) : dernière ``Conv2d`` via
      :func:`last_conv_layer`.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "forward_features"):
        return last_spatial_layer(backbone, probe_shape)
    return last_conv_layer(model)


def resolve_layer(model: nn.Module, name: str) -> nn.Module:
    """Renvoie le sous-module nommé ``name`` (cf. ``model.named_modules()``)."""
    modules = dict(model.named_modules())
    if name not in modules:
        available = ", ".join(n for n in modules if n)
        raise KeyError(f"couche {name!r} introuvable. Disponibles : {available}")
    return modules[name]


class GradCAM:
    """Grad-CAM sur une couche cible d'un modèle.

    Réutilisable : chaque appel à :meth:`compute` refait forward + backward et
    écrase les activations / gradients mémorisés par les hooks.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """Enregistre les hooks avant/arrière sur ``target_layer``."""
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._capture_activations)
        target_layer.register_full_backward_hook(self._capture_gradients)

    def _capture_activations(self, _module, _inputs, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _capture_gradients(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def compute(
        self, input_tensor: torch.Tensor, target_index: int | None = None
    ) -> npt.NDArray[np.float32]:
        """Carte Grad-CAM ``(H, W)`` en float32 dans ``[0, 1]``.

        Args:
            input_tensor: Batch d'une image, ``(1, C, H, W)``.
            target_index: Indice de classe à expliquer ; ``None`` = classe prédite
                (ignoré si le modèle n'a qu'une sortie, cas binaire).
        """
        self.model.zero_grad()
        # requires_grad sur l'entrée : force l'autograd à propager jusqu'à la
        # couche cible même quand tout l'amont est gelé (backbone freeze="all"),
        # sinon le hook arrière ne reçoit jamais de gradient.
        input_tensor = input_tensor.detach().requires_grad_(True)
        logits = self.model(input_tensor)

        if logits.shape[1] == 1:
            score = logits[0, 0]
        else:
            if target_index is None:
                target_index = int(logits[0].argmax())
            score = logits[0, target_index]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]

        cam = cam - cam.min()
        peak = cam.max()
        if peak > 0:
            cam = cam / peak
        return cam.cpu().numpy().astype(np.float32)


def render(
    image_path: str | Path,
    cam: npt.NDArray[np.float32],
    out_dir: str | Path,
    *,
    alpha: float = 0.4,
    cmap: str = "jet",
) -> dict[str, Path]:
    """Sauvegarde les vues Grad-CAM d'une image et renvoie ``{nom: chemin}``.

    Quatre fichiers dans ``out_dir`` (créé si besoin) : les trois vues séparées
    ``<stem>_{original,heatmap,overlay}.png`` et une figure combinée
    ``<stem>_gradcam.png`` (``panel``) — les trois côte à côte avec une barre de
    couleurs. L'image est redimensionnée (resize + pad) à la taille de ``cam``
    pour que les vues se superposent.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    h, w = cam.shape
    image = resize_with_pad(load_image_rgb(image_path), (h, w), fill=255)  # CHW uint8
    image = image.permute(1, 2, 0).numpy() / 255.0  # HWC float [0, 1]

    heatmap = plt.get_cmap(cmap)(cam)[..., :3]  # HWC float [0, 1]
    overlay = np.clip((1.0 - alpha) * image + alpha * heatmap, 0.0, 1.0)

    views = (("original", image), ("heatmap", heatmap), ("overlay", overlay))
    written: dict[str, Path] = {}
    for name, array in views:
        path = out_dir / f"{stem}_{name}.png"
        plt.imsave(path, array)
        written[name] = path

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, (title, array) in zip(axes, (("Original", image), ("Heatmap", heatmap), ("Superposition", overlay))):
        ax.imshow(array)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("activation Grad-CAM")
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["faible", "élevée"])

    panel_path = out_dir / f"{stem}_gradcam.png"
    fig.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written["panel"] = panel_path
    return written
