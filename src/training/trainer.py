"""Boucle d'entraînement / validation / évaluation, isolée du modèle."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.metrics import CURVE_METRICS

if TYPE_CHECKING:
    from torch import nn
    from torchmetrics import MetricCollection

    from src.config import TrainConfig


def _flatten_metric(name: str, value, prefix: str = "test") -> float | list | dict | None:
    """Aplati une valeur de ``MetricCollection.compute()`` pour le dict de sortie d'``evaluate``.

    - tenseur scalaire -> ``float`` ;
    - tenseur vecteur / matrice -> ``list`` imbriquée ;
    - triplet ``(precision, recall, thresholds)`` d'une métrique listée dans
      ``CURVE_METRICS`` -> ``dict`` de ces trois listes ;
    - tout autre tuple (métrique inconnue, ou pas un triplet) -> ``None``, la clé
      est alors exclue du dict (on ne devine pas un format non reconnu).

    ``prefix`` doit correspondre au préfixe utilisé par ``evaluate`` (``test``
    par défaut, ``val`` pour une évaluation sur le jeu de validation).
    """
    if isinstance(value, tuple):
        if name.removeprefix(f"{prefix}_") in CURVE_METRICS and len(value) == 3:
            precision, recall, thresholds = value
            return {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": thresholds.tolist(),
            }
        return None
    return value.item() if value.numel() == 1 else value.tolist()


class Trainer:
    """Entraîne un ``nn.Module`` : boucle epoch, scheduler, callbacks, métriques."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        criterion,
        *,
        optimizer,
        scheduler=None,
        metrics: MetricCollection | None = None,
        callbacks=(),
        device: str | torch.device = "cpu",
        train_sampler=None,
    ):
        """Configure le modèle, l'optimiseur, le scheduler, les callbacks et les métriques.

        ``train_sampler`` : sampler PyTorch optionnel pour le ``DataLoader``
        d'entraînement (ex. ``WeightedRandomSampler``). S'il est fourni, le
        mélange aléatoire (``shuffle``) est désactivé, le sampler s'en charge.
        """
        self.model = model
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_sampler = train_sampler
        self.callbacks = list(callbacks)
        self.device = torch.device(device)
        self.train_metrics = metrics.clone(prefix="train_") if metrics is not None else None
        self.val_metrics = metrics.clone(prefix="val_") if metrics is not None else None
        self.test_metrics = None

    def fit(self, train_ds, val_ds=None) -> dict:
        """Entraîne le modèle et renvoie l'historique des métriques par epoch.

        Args:
            train_ds: jeu de données d'entraînement.
            val_ds: jeu de données de validation, requis si un callback ou un
                ``ReduceLROnPlateau`` est configuré.

        Returns:
            dict: historique ``{clé: [valeur par epoch]}`` (``train_loss``, ``lr``,
            métriques ``train_*`` et, si validation, ``val_loss`` et ``val_*``).

        Raises:
            ValueError: si ``val_ds`` est absent alors qu'une validation est nécessaire.
        """
        self.model.to(self.device)
        if self.train_metrics is not None:
            self.train_metrics = self.train_metrics.to(self.device)
            self.val_metrics = self.val_metrics.to(self.device)

        is_plateau = isinstance(self.scheduler, ReduceLROnPlateau)
        if val_ds is None and (is_plateau or self.callbacks):
            raise ValueError(
                "early_stopping et ReduceLROnPlateau ont besoin d'un jeu de validation."
            )

        persistent_workers = self.config.num_workers > 0
        train_loader = DataLoader(
            train_ds,
            self.config.batch_size,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,
            num_workers=self.config.num_workers,
            persistent_workers=persistent_workers,
        )
        val_loader = (
            DataLoader(
                val_ds,
                self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                persistent_workers=persistent_workers,
            )
            if val_ds is not None
            else None
        )

        # Attribut (pas variable locale) : reste lisible via trainer.history si
        # une KeyboardInterrupt coupe fit() avant son `return`.
        self.history: dict = defaultdict(list)
        try:
            for epoch in range(self.config.nb_epoch):
                with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}: ", unit="batch") as bar:
                    metrics_train = self._train_epoch(train_loader, bar)
                    self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
                    metrics_val = self._validate_epoch(val_loader, bar, metrics_train) if val_loader else {}

                epoch_metrics = {**metrics_train, **metrics_val}
                for key, value in epoch_metrics.items():
                    self.history[key].append(value)

                if self.scheduler is not None:
                    if is_plateau:
                        self.scheduler.step(epoch_metrics["val_loss"])
                    else:
                        self.scheduler.step()

                if any(cb.step(epoch_metrics, self.model) for cb in self.callbacks):
                    print(f"Arrêt anticipé à l'epoch {epoch + 1}")
                    break
        except KeyboardInterrupt:
            print("\nEntraînement interrompu (Ctrl+C) — restauration du meilleur modèle disponible.")
            raise
        finally:
            # Recharge les poids de la meilleure epoch (si un callback les a suivis),
            # que la boucle se termine normalement ou soit interrompue.
            for cb in self.callbacks:
                cb.restore(self.model)
        return dict(self.history)

    def predict(self, x):
        """Renvoie les sorties brutes du modèle pour l'entrée ``x``."""
        return self.model(x)

    def _train_epoch(self, loader, bar) -> dict:
        self.model.train()
        n_samples, running = 0, 0.0
        out = OrderedDict({"train_loss": 0.0})
        for x, y in loader:
            bar.update(1)
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(x)
            loss = self.criterion(preds, y)
            loss.backward()
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
            self.optimizer.step()

            bs = x.size(0)
            n_samples += bs
            running += loss.item() * bs
            out["train_loss"] = running / n_samples
            if self.train_metrics is not None:
                self.train_metrics.update(preds.detach(), y)
                out.update({k: v.item() for k, v in self.train_metrics.compute().items()})
            bar.set_postfix(out)
        if self.train_metrics is not None:
            self.train_metrics.reset()
        return dict(out)

    @torch.no_grad()
    def _validate_epoch(self, loader, bar, metrics_train) -> dict:
        self.model.eval()
        n_samples, running = 0, 0.0
        out = OrderedDict({"val_loss": 0.0})
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x)
            loss = self.criterion(preds, y)
            bs = x.size(0)
            n_samples += bs
            running += loss.item() * bs
            out["val_loss"] = running / n_samples
            if self.val_metrics is not None:
                self.val_metrics.update(preds.detach(), y)
                out.update({k: v.item() for k, v in self.val_metrics.compute().items()})
            bar.set_postfix({**metrics_train, **out})
        if self.val_metrics is not None:
            self.val_metrics.reset()
        return dict(out)

    @torch.no_grad()
    def evaluate(
        self, dataset, metrics: MetricCollection | None = None, prefix: str = "test"
    ) -> dict:
        """Évalue le modèle sur ``dataset`` et renvoie la perte et les métriques ``<prefix>_*``.

        Args:
            dataset: jeu de données à évaluer (test, ou validation pour une
                comparaison de grid search sans toucher au test set).
            metrics: collection de métriques à calculer, préfixée ``<prefix>_``.
            prefix: préfixe des clés du dict renvoyé (``"test"`` par défaut).

        Returns:
            dict: ``{"<prefix>_loss": ..., "<prefix>_<métrique>": ...}``.
        """
        self.model.to(self.device)
        self.model.eval()
        loss_key = f"{prefix}_loss"
        eval_metrics = metrics.clone(prefix=f"{prefix}_").to(self.device) if metrics is not None else None
        loader = DataLoader(
            dataset, self.config.batch_size, shuffle=False, num_workers=self.config.num_workers
        )

        n_samples, running = 0, 0.0
        out = OrderedDict({loss_key: 0.0})
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x)
            loss = self.criterion(preds, y)
            bs = x.size(0)
            n_samples += bs
            running += loss.item() * bs
            out[loss_key] = running / n_samples
            if eval_metrics is not None:
                eval_metrics.update(preds.detach(), y)
        if eval_metrics is not None:
            # ``.compute()`` une seule fois après la boucle : le rebuild de la
            # matrice de confusion / des courbes PR à chaque batch est quadratique.
            out.update(
                {
                    k: flat
                    for k, v in eval_metrics.compute().items()
                    if (flat := _flatten_metric(k, v, prefix=prefix)) is not None
                }
            )
        return dict(out)
