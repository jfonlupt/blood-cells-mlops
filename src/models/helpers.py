"""Helper module."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", save_dir=None):
    """Affiche la matrice de confusion."""
    confusion_matrix = pd.crosstab(
        pd.Series(y_true, name="Actual"),
        pd.Series(y_pred, name="Predicted"),
        rownames=["Actual"],
        colnames=["Predicted"],
    )
    plt.figure(figsize=(10, 8))
    cm = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    cm.set_title(title)
    cm.set_xlabel("Predicted")
    cm.set_ylabel("True")
    if save_dir:
        plt.savefig(save_dir / "confusion_matrix.png")
        plt.close()
    else:
        plt.show()


def plot_train_metrics(history, save_dir=None):
    """Affiche les métriques d'entrainement."""
    train_loss = history["train_loss"]
    val_loss = history.get("val_loss") or None
    _, ax = plt.subplots()
    sns.lineplot(x=range(len(train_loss)), y=train_loss, label="Train", ax=ax)
    if val_loss:
        sns.lineplot(x=range(len(val_loss)), y=val_loss, label="Validation", ax=ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if save_dir:
        plt.savefig(save_dir / "training_loss.png")
        plt.close()
    else:
        plt.show()

    if "lr" in history:
        lr = history["lr"]
        _, ax = plt.subplots()
        sns.lineplot(x=range(len(lr)), y=lr, ax=ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_yscale("log")
        if save_dir:
            plt.savefig(save_dir / "training_lr.png")
            plt.close()
        else:
            plt.show()

    metrics = {k.split("_")[1] for k in history if "_" in k and not k.endswith("_loss")}
    for metric in metrics:
        train_metric = history[f"train_{metric}"]
        val_metric = history.get(f"val_{metric}") or None
        _, ax = plt.subplots()
        sns.lineplot(x=range(len(train_metric)), y=train_metric, label="Train", ax=ax)
        if val_metric:
            sns.lineplot(x=range(len(val_metric)), y=val_metric, label="Validation", ax=ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        if save_dir:
            plt.savefig(save_dir / f"training_{metric}.png")
            plt.close()
        else:
            plt.show()
