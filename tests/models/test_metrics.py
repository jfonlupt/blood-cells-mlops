"""Tests de la fabrique de métriques."""

import pandas as pd
import pytest
import torch
from torchmetrics import MetricCollection

from src.constants import TYPE_TO_LABEL
from src.models.metrics import (
    CURVE_METRICS,
    CriticalScreeningAveragePrecision,
    GroupedRecall,
    build_metrics,
    display_metrics,
    screening_operating_points,
)
from src.tasks import get_task
from src.training.trainer import _flatten_metric

BLAST = TYPE_TO_LABEL["Blast"]
PROMYELOCYTE = TYPE_TO_LABEL["Promyelocyte"]
NEUTROPHIL = TYPE_TO_LABEL["Neutrophil"]
N_TYPES = len(TYPE_TO_LABEL)


def _one_hot_logits(class_indices: list[int], n: int = N_TYPES) -> torch.Tensor:
    """Logits quasi déterministes : ~proba 1 sur la classe demandée."""
    logits = torch.full((len(class_indices), n), -10.0)
    for row, idx in enumerate(class_indices):
        logits[row, idx] = 10.0
    return logits


def test_epoch_metrics_are_lightweight():
    mc = build_metrics(get_task("type"))
    assert isinstance(mc, MetricCollection)
    assert set(mc) == {"acc", "f1"}


def test_test_metrics_are_rich():
    mc = build_metrics(get_task("type"), for_test=True)
    assert "confusion_matrix" in mc
    assert "f1_macro" in mc


def test_test_metrics_include_pr_curve():
    mc = build_metrics(get_task("type"), for_test=True)
    assert "pr_curve" in mc
    assert "pr_curve" in CURVE_METRICS


def test_top3_metrics_present_when_at_least_five_classes():
    mc = build_metrics(get_task("type"), for_test=True)  # 10 classes
    assert {"top3_acc", "top3_f1", "top3_recall_per_class"} <= set(mc)


def test_top3_metrics_absent_for_few_classes():
    mc = build_metrics(get_task("criticality"), for_test=True)  # 3 classes
    assert not any(k.startswith("top3_") for k in mc)


def test_top3_acc_counts_second_ranked_true_class_as_hit():
    mc = build_metrics(get_task("type"), for_test=True)
    logits = torch.full((4, 10), -5.0)
    logits[:, 0] = 5.0  # classe 0 toujours en tête
    logits[:, 1] = 3.0  # vraie classe : 2e rang -> hors top1, dans top3
    mc.update(logits, torch.ones(4, dtype=torch.long))
    computed = mc.compute()
    assert computed["top3_acc"].item() == pytest.approx(1.0)


def test_display_metrics_includes_top3_recall_column():
    n = 10
    raw = {
        "test_recall_per_class": [0.5] * n,
        "test_top3_recall_per_class": [0.9] * n,
        "test_top3_acc": 0.88,
    }
    out = display_metrics(raw, num_classes=n, verbose=False)
    assert "top3_recall" in out["per_class"].columns
    assert out["scalars"]["top3_acc"] == pytest.approx(0.88)


def test_display_metrics_adds_support_from_confusion_matrix():
    raw = {
        "test_recall_per_class": [0.8, 0.6, 0.5],
        "test_confusion_matrix": [[8, 1, 1], [2, 3, 0], [0, 1, 1]],  # sommes de ligne : 10, 5, 2
    }
    out = display_metrics(raw, num_classes=3, verbose=False)
    assert list(out["per_class"]["support"]) == [10, 5, 2]
    assert list(out["per_class"].columns)[-1] == "support"  # dernière colonne


def test_display_metrics_support_excluded_from_bar_chart(tmp_path):
    raw = {
        "test_recall_per_class": [0.8, 0.6, 0.5],
        "test_precision_per_class": [0.7, 0.5, 0.4],
        "test_confusion_matrix": [[8, 1, 1], [2, 3, 0], [0, 1, 1]],
    }
    out = display_metrics(raw, num_classes=3, save_dir=tmp_path, verbose=False)
    ax = out["figures"]["per_class"].axes[0]
    assert ax.get_ylim() == (0.0, 1.0)  # support (comptes) n'écrase pas l'échelle [0,1]
    csv = pd.read_csv(tmp_path / "metrics_test_per_class.csv", index_col=0)
    assert "support" in csv.columns


def test_display_metrics_no_support_without_confusion_matrix():
    out = display_metrics({"test_recall_per_class": [0.8, 0.6, 0.5]}, num_classes=3, verbose=False)
    assert "support" not in out["per_class"].columns


def test_display_metrics_saves_pr_curve_figure(tmp_path):
    n = 10
    raw = {
        "test_pr_curve": {
            "precision": [[1.0, 0.6, 0.0] for _ in range(n)],
            "recall": [[0.0, 0.5, 1.0] for _ in range(n)],
            "thresholds": [0.33, 0.66],
        },
        "test_auprc_per_class": [0.5] * n,
    }
    out = display_metrics(raw, num_classes=n, save_dir=tmp_path, verbose=False)
    assert "pr_curve" in out["figures"]
    assert (tmp_path / "metrics_test_pr_curve.png").exists()


def test_grouped_critical_recall_counts_cross_critical_prediction_as_hit():
    metric = GroupedRecall([BLAST, PROMYELOCYTE])
    # vrai Blast prédit Promyelocyte : reste dans le groupe critique -> compté
    metric.update(_one_hot_logits([PROMYELOCYTE]), torch.tensor([BLAST]))
    assert metric.compute().item() == pytest.approx(1.0)


def test_grouped_critical_recall_counts_critical_predicted_normal_as_miss():
    metric = GroupedRecall([BLAST, PROMYELOCYTE])
    metric.update(_one_hot_logits([NEUTROPHIL]), torch.tensor([BLAST]))
    assert metric.compute().item() == pytest.approx(0.0)


def test_grouped_critical_recall_nan_without_critical_support():
    metric = GroupedRecall([BLAST, PROMYELOCYTE])
    metric.update(_one_hot_logits([NEUTROPHIL]), torch.tensor([NEUTROPHIL]))
    assert torch.isnan(metric.compute())


def test_critical_screening_ap_perfect_on_separable_scores():
    metric = CriticalScreeningAveragePrecision([BLAST, PROMYELOCYTE])
    logits = _one_hot_logits([BLAST, PROMYELOCYTE, NEUTROPHIL, NEUTROPHIL])
    labels = torch.tensor([BLAST, PROMYELOCYTE, NEUTROPHIL, NEUTROPHIL])
    metric.update(logits, labels)
    assert metric.compute().item() == pytest.approx(1.0)


def test_critical_screening_prc_is_a_curve_metric():
    assert "critical_screening_prc" in CURVE_METRICS


def test_build_metrics_type_task_has_criticality_screening():
    mc = build_metrics(get_task("type"), for_test=True)
    assert {
        "grouped_critical_recall",
        "critical_screening_ap",
        "critical_screening_prc",
    } <= set(mc)


def test_build_metrics_criticality_task_has_no_screening():
    mc = build_metrics(get_task("criticality"), for_test=True)
    assert not any("screening" in k or k == "grouped_critical_recall" for k in mc)


def test_flatten_metric_handles_screening_curve():
    triplet = (torch.tensor([1.0, 0.5]), torch.tensor([0.0, 1.0]), torch.tensor([0.33]))
    out = _flatten_metric("test_critical_screening_prc", triplet, prefix="test")
    assert set(out) == {"precision", "recall", "thresholds"}


def test_screening_operating_points_reports_review_load():
    # 2 vrais critiques sur 10 images. À recall 1.0 il faut flaguer les 2 ;
    # si la précision y est de 0.5 -> 4 images flaguées -> charge de revue 0.4.
    precision = [0.5, 1.0]
    recall = [1.0, 0.5]
    thresholds = [0.2, 0.8]
    points = screening_operating_points(
        precision, recall, thresholds, n_critical=2, n_total=10, targets=(0.95,)
    )
    assert points[0.95]["review_load"] == pytest.approx(0.4)
    assert points[0.95]["threshold"] == pytest.approx(0.2)


def test_criticality_screening_survives_collection_clone_and_flatten():
    # Reproduit le chemin Trainer.evaluate : clone(prefix=), update(logits, labels),
    # compute(), puis _flatten_metric sur chaque sortie.
    mc = build_metrics(get_task("type"), for_test=True).clone(prefix="test_")
    logits = _one_hot_logits([BLAST, PROMYELOCYTE, NEUTROPHIL, NEUTROPHIL])
    labels = torch.tensor([BLAST, PROMYELOCYTE, NEUTROPHIL, NEUTROPHIL])
    mc.update(logits, labels)
    flat = {
        k: _flatten_metric(k, v, prefix="test")
        for k, v in mc.compute().items()
        if _flatten_metric(k, v, prefix="test") is not None
    }
    assert flat["test_grouped_critical_recall"] == pytest.approx(1.0)
    assert flat["test_critical_screening_ap"] == pytest.approx(1.0)
    assert set(flat["test_critical_screening_prc"]) == {"precision", "recall", "thresholds"}


def test_display_metrics_reports_grouped_critical_recall_and_screening_figure(tmp_path):
    n = N_TYPES
    raw = {
        "test_grouped_critical_recall": 0.97,
        "test_critical_screening_ap": 0.91,
        "test_critical_screening_prc": {
            "precision": [0.2, 0.5, 1.0],
            "recall": [1.0, 0.8, 0.3],
            "thresholds": [0.25, 0.5, 0.75],
        },
        "test_confusion_matrix": [[10 if i == j else 0 for j in range(n)] for i in range(n)],
    }
    out = display_metrics(raw, num_classes=n, save_dir=tmp_path, verbose=False)
    assert out["scalars"]["grouped_critical_recall"] == pytest.approx(0.97)
    assert out["scalars"]["critical_screening_ap"] == pytest.approx(0.91)
    assert "critical_screening" in out["figures"]
    assert (tmp_path / "metrics_test_critical_screening.png").exists()


def test_binary_task_metrics():
    mc = build_metrics(get_task("dataset"))
    # seuil 0.5 : 0.9 -> 1 (ok), 0.1 -> 0 (ok), 0.2 -> 0 (faux, cible 1) => 2/3
    preds = torch.tensor([0.9, 0.1, 0.2])
    target = torch.tensor([1, 0, 1])
    mc.update(preds, target)
    assert mc.compute()["acc"].item() == pytest.approx(2 / 3)
