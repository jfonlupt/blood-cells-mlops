"""Tests des callbacks d'entrainement."""

import pytest
import torch
from torch import nn

from src.training.callbacks import EarlyStopping


def test_step_returns_false_while_improving():
    """Tant que la métrique s'améliore, l'entrainement continue."""
    es = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    model = nn.Linear(2, 1)

    assert es.step({"val_loss": 1.0}, model) is False
    assert es.step({"val_loss": 0.5}, model) is False
    assert es.step({"val_loss": 0.1}, model) is False


def test_step_stops_after_patience_epochs_without_improvement():
    """Après `patience` epochs sans amélioration, step renvoie True."""
    es = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    model = nn.Linear(2, 1)

    assert es.step({"val_loss": 1.0}, model) is False  # best
    assert es.step({"val_loss": 1.1}, model) is False  # 1 sans amélioration
    assert es.step({"val_loss": 1.2}, model) is True  # 2 sans amélioration -> stop
    assert es.stopped_epoch == 2


def test_tracks_best_score_and_epoch():
    """best_score et best_epoch pointent sur la meilleure epoch."""
    es = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    model = nn.Linear(2, 1)

    es.step({"val_loss": 1.0}, model)  # epoch 0
    es.step({"val_loss": 0.3}, model)  # epoch 1 -> best
    es.step({"val_loss": 0.8}, model)  # epoch 2

    assert es.best_score == pytest.approx(0.3)
    assert es.best_epoch == 1


def test_min_delta_ignores_marginal_improvement():
    """Une amélioration inférieure à min_delta ne compte pas."""
    es = EarlyStopping(monitor="val_loss", mode="min", patience=1, min_delta=0.1)
    model = nn.Linear(2, 1)

    assert es.step({"val_loss": 1.0}, model) is False
    assert es.step({"val_loss": 0.95}, model) is True  # gain 0.05 < 0.1 -> pas d'amélioration


def test_mode_max_treats_higher_as_better():
    """En mode 'max', une métrique plus haute est une amélioration."""
    es = EarlyStopping(monitor="val_acc", mode="max", patience=1)
    model = nn.Linear(2, 1)

    assert es.step({"val_acc": 0.7}, model) is False
    assert es.step({"val_acc": 0.8}, model) is False  # amélioration
    assert es.step({"val_acc": 0.75}, model) is True  # régression -> stop


def test_restore_reloads_best_weights():
    """restore recharge les poids de la meilleure epoch."""
    es = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
    model = nn.Linear(2, 1)

    with torch.no_grad():
        model.weight.fill_(1.0)
    es.step({"val_loss": 0.1}, model)  # snapshot des poids à 1.0
    best_weight = model.weight.clone()

    with torch.no_grad():
        model.weight.fill_(9.0)
    es.step({"val_loss": 0.5}, model)  # pire, pas de snapshot

    es.restore(model)
    assert torch.equal(model.weight, best_weight)


def test_restore_is_noop_when_disabled():
    """Sans restore_best_weights, les poids courants sont conservés."""
    es = EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=False)
    model = nn.Linear(2, 1)

    with torch.no_grad():
        model.weight.fill_(1.0)
    es.step({"val_loss": 0.1}, model)
    with torch.no_grad():
        model.weight.fill_(9.0)
    es.step({"val_loss": 0.5}, model)

    es.restore(model)
    assert torch.equal(model.weight, torch.full_like(model.weight, 9.0))


def test_missing_monitor_key_raises():
    """Une clé de monitoring absente lève une erreur explicite."""
    es = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    model = nn.Linear(2, 1)

    with pytest.raises(KeyError, match="val_loss"):
        es.step({"train_loss": 1.0}, model)
