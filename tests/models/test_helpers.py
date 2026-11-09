"""Tests des helpers de visualisation d'entrainement."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.models.helpers import plot_train_metrics


def test_plot_train_metrics_handles_lr_history(tmp_path):
    """Un historique contenant 'lr' produit une courbe dédiée sans planter."""
    history = {
        "train_loss": [1.0, 0.8, 0.6],
        "val_loss": [1.1, 0.9, 0.7],
        "train_acc": [0.4, 0.5, 0.6],
        "val_acc": [0.3, 0.45, 0.55],
        "lr": [0.01, 0.01, 0.001],
    }

    plot_train_metrics(history, save_dir=tmp_path)

    assert (tmp_path / "training_loss.png").exists()
    assert (tmp_path / "training_acc.png").exists()
    assert (tmp_path / "training_lr.png").exists()


def test_plot_train_metrics_lr_uses_log_scale(tmp_path, monkeypatch):
    """Le subplot du learning rate est tracé en échelle log."""
    axes = []
    original_subplots = plt.subplots

    def _tracking_subplots(*a, **k):
        fig, ax = original_subplots(*a, **k)
        axes.append(ax)
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _tracking_subplots)

    history = {
        "train_loss": [1.0, 0.8],
        "lr": [0.01, 0.001],
    }
    plot_train_metrics(history, save_dir=tmp_path)

    lr_ax = axes[1]  # axes[0] = loss, axes[1] = lr
    assert lr_ax.get_yscale() == "log"


def test_plot_train_metrics_without_validation(tmp_path):
    """Un historique sans clés ``val_*`` (validation='none') ne plante pas."""
    history = {
        "train_loss": [1.0, 0.8, 0.6],
        "train_acc": [0.4, 0.5, 0.6],
        "lr": [0.01, 0.01, 0.001],
    }

    plot_train_metrics(history, save_dir=tmp_path)

    assert (tmp_path / "training_loss.png").exists()
    assert (tmp_path / "training_acc.png").exists()
    assert (tmp_path / "training_lr.png").exists()
