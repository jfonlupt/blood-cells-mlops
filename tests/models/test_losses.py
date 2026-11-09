"""Tests de la perte unique (ClinicalLoss), de ses cas limites et de build_loss."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.config import LossConfig
from src.models.losses import ClinicalLoss, build_loss, effective_number_weights
from src.tasks import get_task


def _df(labels):
    return pd.DataFrame({"label": labels, "type": ["x"] * len(labels)})


# --- effective_number_weights : bornes de beta -----------------------------


def test_effective_number_weights_beta_zero_is_uniform():
    w = effective_number_weights([10, 100, 1000], beta=0.0)
    assert np.allclose(w, 1.0)


def test_effective_number_weights_beta_one_is_inverse_frequency():
    counts = np.array([10.0, 100.0, 1000.0])
    w = effective_number_weights(counts, beta=1.0)
    inv = (1.0 / counts)
    inv = inv / inv.sum() * len(inv)
    assert np.allclose(w, inv)


def test_effective_number_weights_mean_is_one():
    w = effective_number_weights([5, 50, 500, 5000], beta=0.999)
    assert w.mean() == pytest.approx(1.0)


# --- ClinicalLoss : cas particuliers --------------------------------------


def _logits_target(n_classes, batch=16, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, n_classes), torch.randint(0, n_classes, (batch,))


def test_reduces_to_cross_entropy_when_all_terms_off():
    logits, target = _logits_target(4)
    clinical = ClinicalLoss([1, 1, 1, 1], beta=0.0, gamma=0.0, lambda_cost=0.0)
    ce = nn.functional.cross_entropy(logits, target)
    assert clinical(logits, target).item() == pytest.approx(ce.item(), rel=1e-5)


def test_beta_weights_the_cross_entropy():
    logits, target = _logits_target(3)
    counts = [100, 100, 5]
    plain = ClinicalLoss(counts, beta=0.0)(logits, target).item()
    weighted = ClinicalLoss(counts, beta=0.999)(logits, target).item()
    assert weighted != pytest.approx(plain)


def test_gamma_recovers_focal_downweighting():
    # exemple plutôt facile (bien classé mais pas saturé) -> focal réduit sa contribution
    logits = torch.tensor([[2.0, -1.0, -1.0]])
    target = torch.tensor([0])
    base = ClinicalLoss([1, 1, 1], beta=0.0, gamma=0.0)(logits, target).item()
    focal = ClinicalLoss([1, 1, 1], beta=0.0, gamma=2.0)(logits, target).item()
    assert focal < base


def test_lambda_cost_adds_expected_cost_term():
    logits, target = _logits_target(3)
    m = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [10.0, 1.0, 0.0]])
    without = ClinicalLoss([1, 1, 1], beta=0.0, lambda_cost=0.0, cost_matrix=m)(logits, target)
    with_cost = ClinicalLoss([1, 1, 1], beta=0.0, lambda_cost=1.0, cost_matrix=m)(logits, target)
    assert with_cost > without


def test_is_differentiable():
    logits, target = _logits_target(10)
    logits = logits.requires_grad_(True)
    m = 1.0 - torch.eye(10)
    ClinicalLoss(list(range(1, 11)), beta=0.999, gamma=2.0, lambda_cost=0.5, cost_matrix=m)(
        logits, target
    ).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_extra_weight_boosts_named_classes():
    loss = ClinicalLoss([100, 100, 100], class_names=["a", "b", "c"], beta=0.0,
                        extra_weight={"b": 3.0})
    w = loss.class_weights
    assert w[1] > w[0] and w[1] > w[2]


# --- build_loss ----------------------------------------------------------


def test_build_loss_default_is_clinical_loss():
    loss = build_loss(LossConfig(), get_task("type"), _df(list(range(10)) * 3))
    assert isinstance(loss, ClinicalLoss)
    assert loss.class_weights.shape == (10,)


def test_build_loss_dataset_task_is_bce():
    loss = build_loss(LossConfig(), get_task("dataset"), _df([0, 1, 0, 1]))
    assert isinstance(loss, nn.BCEWithLogitsLoss)


def test_build_loss_handles_missing_class_in_train_df():
    loss = build_loss(LossConfig(), get_task("type"), _df([0, 0, 1, 2, 2, 2]))
    assert loss.class_weights.shape == (10,)
    assert torch.isfinite(loss.class_weights).all()


# --- lift de la matrice de coût via la criticité ------------------------


def test_cost_matrix_lifted_from_3x3_to_10_classes():
    from src.constants import CRITICALITY, CRITICALITY_TO_LABEL, FINAL_CLASSES

    m3 = [[0.0, 0.5, 1.0], [4.0, 0.0, 1.0], [10.0, 3.0, 0.0]]
    loss = build_loss(
        LossConfig(lambda_cost=1.0, cost_matrix=m3), get_task("type"), _df(list(range(10)) * 3)
    )
    m10 = loss.cost_matrix
    assert m10.shape == (10, 10)
    for i, ci in enumerate(FINAL_CLASSES):
        for j, cj in enumerate(FINAL_CLASSES):
            ti, tj = CRITICALITY_TO_LABEL[CRITICALITY[ci]], CRITICALITY_TO_LABEL[CRITICALITY[cj]]
            assert m10[i, j].item() == pytest.approx(m3[ti][tj])


def test_cost_matrix_zero_within_criticality_block():
    from src.constants import CRITICALITY, FINAL_CLASSES

    loss = build_loss(
        LossConfig(lambda_cost=1.0, cost_matrix=[[0.0, 0.5, 1.0], [4.0, 0.0, 1.0], [10.0, 3.0, 0.0]]),
        get_task("type"), _df(list(range(10)) * 3),
    )
    blast = FINAL_CLASSES.index("Blast")
    promy = FINAL_CLASSES.index("Promyelocyte")
    assert CRITICALITY["Blast"] == CRITICALITY["Promyelocyte"] == "critique"
    assert loss.cost_matrix[blast, promy].item() == 0.0


def test_cost_matrix_2x2_lifted_to_10_classes():
    loss = build_loss(
        LossConfig(lambda_cost=1.0, cost_matrix=[[0.0, 1.0], [10.0, 0.0]]),
        get_task("type"), _df(list(range(10)) * 3),
    )
    assert loss.cost_matrix.shape == (10, 10)


def test_cost_matrix_used_directly_when_task_dimension():
    m3 = [[0.0, 0.5, 1.0], [4.0, 0.0, 1.0], [10.0, 3.0, 0.0]]
    loss = build_loss(LossConfig(lambda_cost=1.0, cost_matrix=m3), get_task("criticality"),
                      _df([0, 1, 2] * 4))
    assert loss.cost_matrix.shape == (3, 3)
    assert torch.equal(loss.cost_matrix, torch.tensor(m3))


def test_cost_matrix_wrong_dimension_raises():
    m4 = [[0.0] * 4 for _ in range(4)]
    with pytest.raises(ValueError, match="cost_matrix"):
        build_loss(LossConfig(lambda_cost=1.0, cost_matrix=m4), get_task("type"),
                   _df(list(range(10)) * 3))


def test_default_cost_matrix_built_when_none_and_lambda_positive():
    loss = build_loss(LossConfig(lambda_cost=1.0), get_task("type"), _df(list(range(10)) * 3))
    assert loss.cost_matrix is not None and loss.cost_matrix.shape == (10, 10)
