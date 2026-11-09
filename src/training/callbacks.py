"""Callbacks d'entrainement pour le ``Trainer``."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


class EarlyStopping:
    """Arrête l'entrainement quand une métrique cesse de s'améliorer.

    Args:
        monitor: clé de la métrique à surveiller dans le dict d'epoch (ex: "val_loss").
        mode: "min" si une valeur plus basse est meilleure, "max" sinon.
        patience: nombre d'epochs sans amélioration toléré avant l'arrêt.
        min_delta: amélioration minimale pour être comptée comme telle.
        restore_best_weights: recharge les poids de la meilleure epoch dans `restore`.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        """Configure la surveillance de la métrique et l'arrêt anticipé."""
        if mode not in ("min", "max"):
            raise ValueError(f"mode doit être 'min' ou 'max', reçu {mode!r}")
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None
        self._best_state: dict | None = None
        self._epoch = -1
        self._wait = 0

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, epoch_metrics: dict, model: nn.Module) -> bool:
        """Enregistre les métriques d'une epoch et indique s'il faut arrêter."""
        self._epoch += 1
        if self.monitor not in epoch_metrics:
            raise KeyError(
                f"Métrique '{self.monitor}' absente des métriques d'epoch "
                f"({sorted(epoch_metrics)}). Une validation est-elle activée ?"
            )
        score = epoch_metrics[self.monitor]

        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = self._epoch
            self._wait = 0
            if self.restore_best_weights:
                self._best_state = deepcopy(model.state_dict())
            return False

        self._wait += 1
        if self._wait >= self.patience:
            self.stopped_epoch = self._epoch
            return True
        return False

    def restore(self, model: nn.Module) -> None:
        """Recharge les poids de la meilleure epoch si l'option est active."""
        if self.restore_best_weights and self._best_state is not None:
            print(f"Restauration du modèle de l'epoch {self.best_epoch + 1}")
            model.load_state_dict(self._best_state)
