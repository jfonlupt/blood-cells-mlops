"""Modèle par transfer learning : backbone timm pré-entraîné + tête de classification.

Le backbone est instancié via ``timm.create_model`` (``num_classes=0``,
``global_pool="avg"``) : il renvoie un vecteur de features ``(B, num_features)``.
Une tête MLP (ou un simple ``Linear``) est ajoutée par-dessus.

``freeze`` pilote le fine-tuning :

- ``"none"`` : tout le backbone est entraînable ;
- ``"all"`` : le backbone est gelé (``requires_grad=False``) et maintenu en mode
  ``eval`` même quand le modèle est en mode ``train`` — les statistiques de
  BatchNorm / LayerNorm restent celles du pré-entraînement (*linear probing*) ;
- entier ``N`` : seuls les ``N`` derniers groupes de stages du backbone (d'après
  ``backbone.group_matcher``) restent entraînables, le reste est gelé
  (``requires_grad=False``). Les BatchNorm des couches gelées **continuent** de
  mettre à jour leurs statistiques (mode ``train`` conservé) — seul ``"all"``
  fige aussi les stats.

``backbone_lr_mult`` permet un taux d'apprentissage discriminatif : voir
:meth:`TransferModel.param_groups`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from src.config import TransferConfig


def _build_head(
    in_features: int, hidden: list[int], num_classes: int, dropout: float
) -> nn.Sequential:
    """Empile ``Linear -> ReLU -> Dropout`` pour chaque taille de ``hidden``, puis la sortie.

    ``hidden == []`` -> un seul ``Linear(in_features, num_classes)``.
    """
    layers: list[nn.Module] = []
    dim = in_features
    for width in hidden:
        layers += [nn.Linear(dim, width), nn.ReLU(), nn.Dropout(dropout)]
        dim = width
    layers.append(nn.Linear(dim, num_classes))
    return nn.Sequential(*layers)


class TransferModel(nn.Module):
    """Backbone timm + tête ; gel optionnel et LR discriminatif."""

    def __init__(self, cfg: TransferConfig, num_classes: int):
        """Instancie le backbone pré-entraîné et la tête.

        Args:
            cfg: Configuration ``model.transfer``.
            num_classes: Nombre de classes de la tâche.
        """
        super().__init__()
        import timm  # import différé : ne pas payer timm quand ce modèle n'est pas utilisé

        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.head = _build_head(
            self.backbone.num_features, list(cfg.head_hidden), num_classes, cfg.head_dropout
        )
        self._backbone_frozen = cfg.freeze == "all"
        self.backbone_lr_mult = cfg.backbone_lr_mult
        if self._backbone_frozen:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        elif isinstance(cfg.freeze, int):
            self._freeze_early_groups(cfg.freeze)

    def _freeze_early_groups(self, n_trainable: int) -> None:
        """Gèle le backbone sauf ses ``n_trainable`` derniers groupes de stages.

        Les groupes sont ceux de ``backbone.group_matcher`` (0 = stem, puis un
        par étage / bloc).
        """
        try:
            from timm.models._manipulate import group_with_matcher
        except ImportError as exc:  # pragma: no cover - garde-fou version timm
            raise RuntimeError(
                "freeze entier nécessite timm.models._manipulate.group_with_matcher "
                "(API interne timm) — épingler timm, ou utiliser freeze='none'/'all'."
            ) from exc

        matcher = self.backbone.group_matcher(coarse=True)
        grouped = group_with_matcher(
            self.backbone.named_parameters(), matcher, return_values=False
        )
        ordered = sorted(grouped, key=lambda g: (g == float("inf"), g))
        if n_trainable >= len(ordered):
            warnings.warn(
                f"freeze={n_trainable} >= {len(ordered)} groupes du backbone "
                f"{type(self.backbone).__name__} : rien n'est gelé.",
                stacklevel=3,
            )
            return
        trainable_groups = set(ordered[-n_trainable:])
        frozen = {
            name
            for gid, names in grouped.items()
            if gid not in trainable_groups
            for name in names
        }
        for name, param in self.backbone.named_parameters():
            if name in frozen:
                param.requires_grad_(False)

    def forward(self, x):
        """Features du backbone -> tête de classification."""
        return self.head(self.backbone(x))

    def features(self, x):
        """Vecteur de features du backbone, avant la tête de classification."""
        return self.backbone(x)

    def train(self, mode: bool = True) -> TransferModel:
        """Comme ``nn.Module.train`` mais garde le backbone gelé en ``eval``."""
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def param_groups(self, base_lr: float) -> list[dict]:
        """Groupes de paramètres pour l'optimiseur, LR du backbone réduit.

        - backbone (paramètres non gelés) : ``base_lr * backbone_lr_mult`` ;
        - tête : ``base_lr``.

        Les groupes vides (backbone entièrement gelé) sont omis.

        Args:
            base_lr: Taux d'apprentissage de référence (``optim.lr``).

        Returns:
            Liste de dicts ``{"params": [...], "lr": float}``.
        """
        groups: list[dict] = []
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * self.backbone_lr_mult})
        groups.append({"params": list(self.head.parameters()), "lr": base_lr})
        return groups
